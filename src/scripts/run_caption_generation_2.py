import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import hydra
import rootutils
import torch
import wandb
import evaluate
import pandas as pd
from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import set_seed
from trl import SFTTrainer

from src.caption.config import CaptionGenerationConfig
from src.utils import RankedLogger, print_config_tree, instantiate_loggers

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


def prepare_datasets(data_cfg: DictConfig) -> Tuple[DatasetDict, List[dict]]:
    data_files = {
        "train": data_cfg.train_file,
        "validation": data_cfg.validation_file,
        "test": data_cfg.test_file,
    }
    raw_dataset = load_dataset("csv", data_files=data_files)

    prompt_cfg = data_cfg.prompt
    text_column = data_cfg.text_column
    remove_columns = data_cfg.remove_columns

    def _format_row(row: Dict[str, Any]) -> Dict[str, str]:
        prompt_template = prompt_cfg.template
        prompt_filled = prompt_template.format(**row)
        return {text_column: prompt_filled}

    processed_dataset = raw_dataset.map(
        _format_row,
        remove_columns=remove_columns,
    )

    return processed_dataset

def prepare_model(cfg: DictConfig) -> Tuple[AutoModelForCausalLM, LoraConfig]:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=quant_cfg,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**OmegaConf.to_container(cfg.lora, resolve=True))
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, lora_config

def prepare_trainer(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: DatasetDict,
    lora_config: LoraConfig,
) -> SFTTrainer:
    training_args_dict = OmegaConf.to_container(cfg.trainer, resolve=True)
    training_args_dict = dict(training_args_dict)

    training_args = TrainingArguments(**training_args_dict)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    return trainer

def compute_metrics(
    predictions: List[str],
    references: List[str],
    metric_cfgs: Iterable[DictConfig],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for metric_cfg in metric_cfgs:
        metric = evaluate.load(metric_cfg.name)
        kwargs = metric_cfg.get("kwargs", {}) or {}
        value = metric.compute(
            predictions=predictions,
            references=references,
            **kwargs,
        )
        results[metric_cfg.name] = value
    return results

def generate_caption(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Cast to BFloat16 if model is in bfloat16 mode
    if model.dtype == torch.bfloat16:
        inputs = {k: v.bfloat16() for k, v in inputs.items()}
    model.eval()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def caption_generation(cfg: CaptionGenerationConfig):
    random_state = cfg.random_state
    set_seed(random_state)

    log.info("Instantiating model...")
    model, tokenizer, lora_config = prepare_model(cfg)

    log.info("Loading dataset...")
    dataset = prepare_datasets(cfg.data)

    log.info("Instantiating trainer...")
    trainer = prepare_trainer(cfg, model, tokenizer, dataset, lora_config)

    log.info("Starting training...")
    trainer.train()

    log.info("Evaluating model...")
    predictions = []
    references = []
    for sample in dataset["test"]:
        prompt = sample["prompt"]
        reference = sample["caption"]
        generated_caption = generate_caption(
            model,
            tokenizer,
            prompt,
            max_length=cfg.generation.max_length,
            temperature=cfg.generation.temperature,
        )
        predictions.append(generated_caption)
        references.append(reference)

    log.info("Computing metrics...")
    metrics = compute_metrics(
        predictions,
        references,
        cfg.evaluation.metrics,
    )

    log.info(f"Evaluation results: {metrics}")
    return metrics