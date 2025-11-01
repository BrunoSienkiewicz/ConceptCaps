from __future__ import annotations

import torch
from datasets import DatasetDict
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from src.caption.logging_utils import WandbMonitoringCallback


def create_trainer(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: DatasetDict,
    lora_config: LoraConfig,
) -> SFTTrainer:
    training_args_dict = OmegaConf.to_container(cfg.trainer, resolve=True)
    training_args_dict = dict(training_args_dict)

    if training_args_dict.get("fp16") is None:
        training_args_dict["fp16"] = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    if training_args_dict.get("bf16") is None:
        training_args_dict["bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(**training_args_dict)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.add_callback(WandbMonitoringCallback())
    return trainer
