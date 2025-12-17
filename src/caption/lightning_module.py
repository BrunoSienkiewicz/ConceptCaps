from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import lightning as pl
import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import EvalPrediction
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.caption.evaluation import MetricComputer, generate_batch_caption_tokenized, generate_caption_tokenized
from src.caption.modeling import build_quantization_config, prepare_tokenizer
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class CaptionFineTuningModule(pl.LightningModule):
    """PyTorch Lightning module for caption fine-tuning."""

    def __init__(
        self,
        model_cfg: DictConfig,
        generation_cfg: DictConfig,
        lora_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        lr_scheduler_cfg: DictConfig,
        prompt_cfg: DictConfig,
        tokenizer: AutoTokenizer,
        metric_computer: Optional[MetricComputer] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "metric_computer"])

        self.model_cfg = model_cfg
        self.generation_cfg = generation_cfg
        self.lora_cfg = lora_cfg
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.prompt_cfg = prompt_cfg
        self.tokenizer = tokenizer
        self.metric_computer = metric_computer
        
        # Initialize model
        self.model = self._setup_model()

    def _setup_model(self) -> AutoModelForCausalLM:
        """Initialize and prepare the model with LoRA."""
        quantization_config = build_quantization_config(self.model_cfg)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.name,
            quantization_config=quantization_config,
            device_map=self.model_cfg.device_map,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        
        if self.model_cfg.checkpoint_dir:
            log.info(f"Loading model weights from checkpoint: {self.model_cfg.checkpoint_dir}...")
            model = PeftModel.from_pretrained(
                model,
                self.model_cfg.checkpoint_dir,
                device_map=self.model_cfg.device_map,
                low_cpu_mem_usage=True,
            )
            return model
        
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        if getattr(self.model_cfg, 'gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        
        # Apply LoRA
        lora_config = LoraConfig(**OmegaConf.to_container(self.lora_cfg, resolve=True))
        model = get_peft_model(model, lora_config)
        
        log.info("Model prepared with LoRA:")
        model.print_trainable_parameters()
        
        return model

    def _extract_prompt_from_batch(self, batch_input_ids, batch_attention_mask):
        """Extract only prompt tokens (up to delimiter) from full sequence."""
        prompt_delimiter = self.prompt_cfg.prompt_delimiter.strip()
        delimiter_token_ids = self.tokenizer.encode(prompt_delimiter, add_special_tokens=False)
        
        prompt_ids = []
        prompt_masks = []
        
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            token_list = input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids
            
            # Find delimiter position
            for j in range(len(token_list) - len(delimiter_token_ids) + 1):
                if token_list[j:j+len(delimiter_token_ids)] == delimiter_token_ids:
                    # Keep tokens up to and including delimiter
                    prompt_end = j + len(delimiter_token_ids)
                    prompt_ids.append(input_ids[:prompt_end])
                    prompt_masks.append(attention_mask[:prompt_end])
                    break
        
        # Pad prompts to same length
        max_prompt_len = max(len(p) for p in prompt_ids)
        
        padded_prompt_ids = []
        padded_prompt_masks = []
        
        for prompt_id, prompt_mask in zip(prompt_ids, prompt_masks):
            # Pad input_ids with pad_token_id
            pad_len = max_prompt_len - len(prompt_id)
            padded_id = torch.nn.functional.pad(
                prompt_id, 
                (0, pad_len), 
                value=self.tokenizer.pad_token_id
            )
            # Pad attention_mask with 0
            padded_mask = torch.nn.functional.pad(
                prompt_mask, 
                (0, pad_len), 
                value=0
            )
            padded_prompt_ids.append(padded_id)
            padded_prompt_masks.append(padded_mask)
        
        return torch.stack(padded_prompt_ids), torch.stack(padded_prompt_masks)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outputs):
        """Compute metrics at the end of validation epoch."""
        predictions = []
        decoded_references = []
        
        log.info(f"Outputs length: {len(outputs)}")
        log.info(f"Outputs type: {type(outputs)}")
        log.info(f"First output type: {type(outputs[0]) if len(outputs) > 0 else 'N/A'}")
        log.info(f"Sample output keys: {outputs[0].keys() if len(outputs) > 0 else 'N/A'}")

        # Iterate over validation dataloader
        for batch in outputs:
            batch_input_ids = torch.tensor(batch["input_ids"]).to(self.model.device)
            batch_attention_mask = torch.tensor(batch["attention_mask"]).to(self.model.device)
            batch_label_ids = torch.tensor(batch["labels"]).to(self.model.device)

            prompt_ids, prompt_masks = self._extract_prompt_from_batch(batch_input_ids, batch_attention_mask)
            prompts = [self.tokenizer.decode(pid, skip_special_tokens=True) for pid in prompt_ids]

            batch_preds = generate_batch_caption_tokenized(
                model=self.model,
                tokenizer=self.tokenizer,
                input_ids=prompt_ids,
                attention_mask=prompt_masks,
                max_new_tokens=self.generation_cfg.max_new_tokens,
            )
            # remove prompts from predictions
            batch_preds = [pred[len(prompts[i]):].strip() for i, pred in enumerate(batch_preds)]
            predictions.extend(batch_preds)

            # remove masked tokens (-100) for decoding
            batch_label_ids[batch_label_ids == -100] = self.tokenizer.pad_token_id
            batch_refs = self.tokenizer.batch_decode(
                batch_label_ids, skip_special_tokens=True
            )
            # remove prompts from references
            batch_refs = [ref[len(prompts[i]):].strip() for i, ref in enumerate(batch_refs)]
            decoded_references.extend(batch_refs)

        if len(predictions) != len(decoded_references):
            log.warning(
                f"Number of predictions ({len(predictions)}) does not match "
                f"number of references ({len(decoded_references)})."
            )
            min_len = min(len(predictions), len(decoded_references))
            log.info(f"Resizing predictions and references to min length: {min_len}")
            predictions = predictions[:min_len]
            decoded_references = decoded_references[:min_len]
        
        log.info(f"Sample prediction: {predictions[0]}")
        log.info(f"Sample reference: {decoded_references[0]}")

        metrics = self.metric_computer.compute_metrics(
            predictions=predictions,
            references=decoded_references
        )

        exclude_types = (dict, list, str, tuple, set)
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, exclude_types):
                        self.log(f"val/{key}_{sub_key}", sub_value, on_epoch=True, sync_dist=True)
                continue
            if key == "mauve":
                self.log(f"val/{key}", value.mauve, on_epoch=True, sync_dist=True)
                continue
            if not isinstance(value, exclude_types):
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        """Compute metrics at the end of test epoch."""
        predictions = []
        decoded_references = []

        # Iterate over test dataloader
        for batch in outputs:
            # Convert from numpy to torch tensors
            batch_input_ids = torch.tensor(batch["input_ids"]).to(self.model.device)
            batch_attention_mask = torch.tensor(batch["attention_mask"]).to(self.model.device)
            batch_label_ids = torch.tensor(batch["labels"]).to(self.model.device)

            prompt_ids, prompt_masks = self._extract_prompt_from_batch(batch_input_ids, batch_attention_mask)
            prompts = [self.tokenizer.decode(pid, skip_special_tokens=True) for pid in prompt_ids]

            batch_preds = generate_batch_caption_tokenized(
                model=self.model,
                tokenizer=self.tokenizer,
                input_ids=prompt_ids,
                attention_mask=prompt_masks,
                max_new_tokens=self.generation_cfg.max_new_tokens,
            )
            # remove prompts from predictions
            batch_preds = [pred[len(prompts[i]):].strip() for i, pred in enumerate(batch_preds)]
            predictions.extend(batch_preds)

            # remove masked tokens (-100) for decoding
            batch_label_ids[batch_label_ids == -100] = self.tokenizer.pad_token_id
            batch_refs = self.tokenizer.batch_decode(
                batch_label_ids, skip_special_tokens=True
            )
            # remove prompts from references
            batch_refs = [ref[len(prompts[i]):].strip() for i, ref in enumerate(batch_refs)]
            decoded_references.extend(batch_refs)

        if len(predictions) != len(decoded_references):
            log.warning(
                f"Number of predictions ({len(predictions)}) does not match "
                f"number of references ({len(decoded_references)})."
            )
            min_len = min(len(predictions), len(decoded_references))
            log.info(f"Resizing predictions and references to min length: {min_len}")
            predictions = predictions[:min_len]
            decoded_references = decoded_references[:min_len]

        log.info(f"Sample prediction: {predictions[0]}")
        log.info(f"Sample reference: {decoded_references[0]}")

        metrics = self.metric_computer.compute_metrics(
            predictions=predictions,
            references=decoded_references
        )

        exclude_types = (dict, list, str, tuple, set)
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, exclude_types):
                        self.log(f"test/{key}_{sub_key}", sub_value, on_epoch=True, sync_dist=True)
                continue
            if key == "mauve":
                self.log(f"test/{key}", value.mauve, on_epoch=True, sync_dist=True)
                continue
            if not isinstance(value, exclude_types):
                self.log(f"test/{key}", value, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        optimizer = AdamW(
            trainable_params,
            lr=self.optimizer_cfg.get("lr", 2e-4),
            weight_decay=self.optimizer_cfg.get("weight_decay", 0.01),
            betas=(
                self.optimizer_cfg.get("adam_beta1", 0.9),
                self.optimizer_cfg.get("adam_beta2", 0.999),
            ),
            eps=self.optimizer_cfg.get("eps", 1e-8),
        )
        
        # Scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.lr_scheduler_cfg.get("T_max", 10),
            eta_min=self.optimizer_cfg.get("eta_min", 0),
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def generate(self, input_ids, attention_mask):
        """Generate captions."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generation_cfg
            )
        return outputs
