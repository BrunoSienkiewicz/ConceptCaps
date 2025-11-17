from __future__ import annotations

from typing import Any, Dict, Optional

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

from src.caption.evaluation import MetricComputer
from src.caption.modeling import build_quantization_config, prepare_tokenizer
from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class CaptionFineTuningModule(pl.LightningModule):
    """PyTorch Lightning module for caption fine-tuning."""

    def __init__(
        self,
        model_cfg: DictConfig,
        lora_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        lr_scheduler_cfg: DictConfig,
        tokenizer: AutoTokenizer,
        metric_computer: Optional[MetricComputer] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "metric_computer"])
        
        self.model_cfg = model_cfg
        self.lora_cfg = lora_cfg
        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.tokenizer = tokenizer
        self.metric_computer = metric_computer
        
        # Initialize model
        if self.model_cfg.checkpoint_dir:
            self.model = self._setup_pretrained_model()
        else:
            self.model = self._setup_model()
        
        # For validation metrics accumulation
        self.validation_step_outputs = []

    def _setup_model(self) -> AutoModelForCausalLM:
        """Initialize and prepare the model with LoRA."""
        quantization_config = build_quantization_config(self.model_cfg)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.name,
            quantization_config=quantization_config,
            device_map=self.model_cfg.device_map,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        if self.model_cfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
        
        # Apply LoRA
        lora_config = LoraConfig(**OmegaConf.to_container(self.lora_cfg, resolve=True))
        model = get_peft_model(model, lora_config)
        
        log.info("Model prepared with LoRA:")
        model.print_trainable_parameters()
        
        return model

    def _setup_pretrained_model(self) -> AutoModelForCausalLM:
        """Load a pretrained model from checkpoint for evaluation."""
        quantization_config = build_quantization_config(self.model_cfg)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.name,
            quantization_config=quantization_config,
            device_map=self.model_cfg.device_map,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )
        
        log.info(f"Loading model weights from checkpoint: {self.model_cfg.checkpoint_dir}...")
        model = PeftModel.from_pretrained(
            model,
            self.model_cfg.checkpoint_dir,
            device_map=self.model_cfg.device_map,
            low_cpu_mem_usage=True,
        )
        model = model.merge_and_unload()
        
        return model

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
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/perplexity", torch.exp(loss), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)
        
        # Store predictions for metric computation
        if self.metric_computer is not None:
            predictions = outputs.logits.argmax(dim=-1)
            self.validation_step_outputs.append({
                "predictions": predictions.detach(),
                "labels": batch["labels"].detach(),
            })
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/perplexity", torch.exp(loss), on_step=False, on_epoch=True, sync_dist=True)

        # Store predictions for metric computation
        if self.metric_computer is not None:
            predictions = outputs.logits.argmax(dim=-1)
            self.validation_step_outputs.append({
                "predictions": predictions.detach(),
                "labels": batch["labels"].detach(),
            })
        
        return loss

    def on_validation_epoch_end(self):
        """Compute metrics at the end of validation epoch."""
        if self.metric_computer is not None and len(self.validation_step_outputs) > 0:
            # Gather all predictions and labels
            all_predictions = torch.cat([x["predictions"] for x in self.validation_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
            
            # Compute metrics
            eval_pred = EvalPrediction(
                predictions=all_predictions.cpu().numpy(),
                label_ids=all_labels.cpu().numpy(),
            )
            
            metrics = self.metric_computer.compute_metrics(eval_pred)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.log(f"val/{key}", value, on_epoch=True, sync_dist=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Compute metrics at the end of test epoch."""
        if self.metric_computer is not None and len(self.validation_step_outputs) > 0:
            # Gather all predictions and labels
            all_predictions = torch.cat([x["predictions"] for x in self.validation_step_outputs])
            all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
            
            # Compute metrics
            eval_pred = EvalPrediction(
                predictions=all_predictions.cpu().numpy(),
                label_ids=all_labels.cpu().numpy(),
            )
            
            metrics = self.metric_computer.compute_metrics(eval_pred)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.log(f"test/{key}", value, on_epoch=True, sync_dist=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()

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

    def generate(self, input_ids, attention_mask, max_new_tokens=256):
        """Generate captions."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return outputs
