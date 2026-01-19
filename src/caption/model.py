from __future__ import annotations

import torch

from typing import Tuple
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from datasets import DatasetDict
from src.caption.evaluation import MetricComputer

from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def build_quantization_config(model_cfg: DictConfig) -> BitsAndBytesConfig | None:
    quant_cfg = model_cfg.get("quantization")
    if not quant_cfg:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
    )


def prepare_tokenizer(model_cfg: DictConfig) -> AutoTokenizer:
    tokenizer_kwargs: dict[str, object] = {}
    tokenizer_cfg = model_cfg.get("tokenizer", {})
    if tokenizer_cfg.get("use_fast") is not None:
        tokenizer_kwargs["use_fast"] = tokenizer_cfg["use_fast"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, **tokenizer_kwargs)
    if tokenizer_cfg.get("padding_side"):
        tokenizer.padding_side = tokenizer_cfg["padding_side"]
    if tokenizer_cfg.get("pad_token_as_eos", True) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_cfg.get("pad_token") is not None:
        tokenizer.pad_token = tokenizer_cfg["pad_token"]
    return tokenizer


def prepare_training_model(model_cfg: DictConfig, lora_cfg: DictConfig) -> AutoModelForCausalLM:
    quantization_config = build_quantization_config(model_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if model_cfg.checkpoint_dir:
        log.info(f"Loading model weights from checkpoint: {model_cfg.checkpoint_dir}...")
        model = PeftModel.from_pretrained(
            model,
            model_cfg.checkpoint_dir,
            device_map=model_cfg.device_map,
            low_cpu_mem_usage=True,
        )
        return model
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**OmegaConf.to_container(lora_cfg, resolve=True))
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


