from __future__ import annotations

from itertools import islice
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

from src.caption.evaluation import MetricComputer, generate_batch_caption_tokenized
from caption.model import prepare_training_model
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

        self.generation_cfg = generation_cfg
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.prompt_cfg = prompt_cfg
        self.tokenizer = tokenizer
        self.metric_computer = metric_computer
        
        self.model = prepare_training_model(model_cfg, lora_cfg)

        # Store predictions and references for epoch-end evaluation
        self.validation_predictions = []
        self.validation_references = []
        self.test_predictions = []
        self.test_references = []

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
            pad_len = max_prompt_len - len(prompt_id)
            padded_id = torch.nn.functional.pad(
                prompt_id, 
                (0, pad_len), 
                value=self.tokenizer.pad_token_id
            )
            padded_mask = torch.nn.functional.pad(
                prompt_mask, 
                (0, pad_len), 
                value=0
            )
            padded_prompt_ids.append(padded_id)
            padded_prompt_masks.append(padded_mask)
        
        return torch.stack(padded_prompt_ids), torch.stack(padded_prompt_masks)

    def _process_batch_for_metrics(self, batch_input_ids, batch_attention_mask, batch_label_ids):
        """
        This method is needed to extract predictions and references for metrics computation.
        During training/validation/test steps, we have full sequences with prompts + targets.
        We need to extract only the generated captions (predictions) and the target captions (references).

        Args:
            batch_input_ids: Tensor of input IDs (prompts + targets)
            batch_attention_mask: Tensor of attention masks
            batch_label_ids: Tensor of label IDs (targets with -100 for prompt tokens)
        """

        prompt_ids, prompt_masks = self._extract_prompt_from_batch(batch_input_ids, batch_attention_mask)
        prompts = [self.tokenizer.decode(pid, skip_special_tokens=True) for pid in prompt_ids]

        # Generate predictions
        batch_preds = generate_batch_caption_tokenized(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prompt_ids,
            attention_mask=prompt_masks,
            max_new_tokens=self.generation_cfg.max_new_tokens,
        )
        # Remove prompts from predictions
        batch_preds = [pred[len(prompts[i]):].strip() for i, pred in enumerate(batch_preds)]

        # Extract references from labels
        # Remove masked tokens (-100) for decoding
        batch_label_ids[batch_label_ids == -100] = self.tokenizer.pad_token_id
        batch_refs = self.tokenizer.batch_decode(
            batch_label_ids, skip_special_tokens=True
        )
        # Remove prompts from references
        batch_refs = [ref[len(prompts[i]):].strip() for i, ref in enumerate(batch_refs)]

        return batch_preds, batch_refs

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
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)

        batch_input_ids = batch["input_ids"].to(self.model.device)
        batch_attention_mask = batch["attention_mask"].to(self.model.device)
        batch_label_ids = batch["labels"].clone().to(self.model.device)
        
        batch_preds, batch_refs = self._process_batch_for_metrics(
            batch_input_ids, batch_attention_mask, batch_label_ids
        )
        
        self.validation_predictions.extend(batch_preds)
        self.validation_references.extend(batch_refs)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)

        batch_input_ids = batch["input_ids"].to(self.model.device)
        batch_attention_mask = batch["attention_mask"].to(self.model.device)
        batch_label_ids = batch["labels"].clone().to(self.model.device)
        
        batch_preds, batch_refs = self._process_batch_for_metrics(
            batch_input_ids, batch_attention_mask, batch_label_ids
        )
        
        self.test_predictions.extend(batch_preds)
        self.test_references.extend(batch_refs)

        return loss

    def on_validation_epoch_end(self):
        """Compute metrics at the end of validation epoch."""
        if len(self.validation_predictions) == 0:
            log.warning("No validation predictions collected.")
            return

        metrics = self.metric_computer.compute_metrics(
            predictions=self.validation_predictions,
            references=self.validation_references
        )

        exclude_types = (dict, list, str, tuple, set)
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, exclude_types):
                        self.log(f"val/{key}_{sub_key}", sub_value, on_epoch=True, sync_dist=True)
                continue
            # Mauve metric uses a special object
            if key == "mauve":
                self.log(f"val/{key}", value.mauve, on_epoch=True, sync_dist=True)
                continue
            if not isinstance(value, exclude_types):
                self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)

        self.validation_predictions = []
        self.validation_references = []

    def on_test_epoch_end(self):
        """Compute metrics at the end of test epoch."""
        if len(self.test_predictions) == 0:
            log.warning("No test predictions collected.")
            return

        metrics = self.metric_computer.compute_metrics(
            predictions=self.test_predictions,
            references=self.test_references
        )

        exclude_types = (dict, list, str, tuple, set)
        
        # Log metrics
        for key, value in metrics.items():
            # Log metric sub-values if dict
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, exclude_types):
                        self.log(f"test/{key}_{sub_key}", sub_value, on_epoch=True, sync_dist=True)
                continue
            # Mauve metric uses a special object
            if key == "mauve":
                self.log(f"test/{key}", value.mauve, on_epoch=True, sync_dist=True)
                continue
            if not isinstance(value, exclude_types):
                self.log(f"test/{key}", value, on_epoch=True, sync_dist=True)

        # Clear stored predictions and references
        self.test_predictions = []
        self.test_references = []

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
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
    
    def generate_caption(self, prompt: str) -> str:
        """Generate caption for a single prompt."""
        self.model.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.generation_cfg.get("max_length", 512),
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **self.generation_cfg
            )
        # Slice off the input tokens to get only generated text
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0, input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def generate_captions_batch(self, prompts: list[str]) -> list[str]:
        """Generate captions for a batch of prompts."""
        self.model.eval()
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.generation_cfg.get("max_length", 512),
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **self.generation_cfg
            )
        # Slice off the input tokens to get only generated text
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_length:]
        batch_captions = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return [caption.strip() for caption in batch_captions]
