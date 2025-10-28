from __future__ import annotations

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

def _flatten_numeric_metrics(value: Any, prefix: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, sub_value in value.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            result.update(_flatten_numeric_metrics(sub_value, new_prefix))
    elif isinstance(value, (int, float)):
        result[prefix] = float(value)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            new_prefix = f"{prefix}/{idx}" if prefix else str(idx)
            result.update(_flatten_numeric_metrics(item, new_prefix))
    return result


class WandbMonitoringCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if logs is None or wandb.run is None:
            return
        payload: Dict[str, float] = {}
        for key, value in logs.items():
            if key == "epoch":
                continue
            if key.startswith("eval_"):
                base = f"eval/{key[5:]}"
            elif key.startswith("train_"):
                base = f"train/{key[6:]}"
            else:
                base = f"train/{key}"
            payload.update(_flatten_numeric_metrics(value, base))
        if state.global_step is not None:
            payload["trainer/global_step"] = state.global_step
        if state.epoch is not None:
            payload["trainer/epoch"] = state.epoch
        if payload:
            wandb.log(payload, step=state.global_step)

def _build_quantization_config(model_cfg: DictConfig) -> BitsAndBytesConfig | None:
    quant_cfg = model_cfg.get("quantization")
    if not quant_cfg:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
    )


def _format_prompt(prompt_cfg: DictConfig, aspects: Any, reference_caption: str) -> str:
    user_prompt = prompt_cfg["user_prompt_template"].format(tags=aspects)
    return prompt_cfg["template"].format(
        system_prompt=prompt_cfg["system_prompt"],
        user_prompt=user_prompt,
        assistant_response=reference_caption,
    ).strip()


def _prepare_datasets(data_cfg: DictConfig) -> Tuple[DatasetDict, List[dict]]:
    data_files = {
        "train": data_cfg.train_file,
        "validation": data_cfg.validation_file,
        "test": data_cfg.test_file,
    }
    raw_dataset = load_dataset("csv", data_files=data_files)

    test_dataset_raw = raw_dataset["test"]
    prompt_cfg = data_cfg.prompt
    text_column = data_cfg.text_column
    remove_columns = data_cfg.remove_columns
    if remove_columns is None:
        remove_columns = raw_dataset["train"].column_names

    def _format_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = _format_prompt(
            prompt_cfg,
            row[data_cfg.aspect_column],
            row[data_cfg.caption_column],
        )
        return {text_column: formatted}

    processed_dataset = raw_dataset.map(
        _format_row,
        remove_columns=remove_columns,
    )

    return processed_dataset, test_dataset_raw.to_list()


def _prepare_tokenizer(model_cfg: DictConfig) -> AutoTokenizer:
    tokenizer_kwargs: Dict[str, Any] = {}
    tokenizer_cfg = model_cfg.get("tokenizer", {})
    if tokenizer_cfg.get("use_fast") is not None:
        tokenizer_kwargs["use_fast"] = tokenizer_cfg["use_fast"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, **tokenizer_kwargs)
    if tokenizer_cfg.get("padding_side"):
        tokenizer.padding_side = tokenizer_cfg["padding_side"]
    if tokenizer_cfg.get("pad_token_as_eos", True) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_model(cfg: DictConfig) -> Tuple[AutoModelForCausalLM, LoraConfig]:
    quantization_config = _build_quantization_config(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        quantization_config=quantization_config,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(**OmegaConf.to_container(cfg.lora, resolve=True))
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config


def _prepare_trainer(
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


def _generate_caption(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _compute_metrics(
    predictions: List[str],
    references: List[str],
    metric_cfgs: Iterable[DictConfig],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for metric_cfg in metric_cfgs:
        metric = evaluate.load(metric_cfg.name)
        kwargs = metric_cfg.get("kwargs", {}) or {}
        if metric_cfg.name == "bleu":
            value = metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references],
                **kwargs,
            )
        else:
            value = metric.compute(
                predictions=predictions,
                references=references,
                **kwargs,
            )
        results[metric_cfg.name] = value
    return results


def _run_evaluation(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_examples: List[dict],
    output_dir: Path,
) -> Dict[str, Any]:
    if not cfg.evaluation.enabled or not eval_examples:
        log.info("Evaluation skipped.")
        return {}

    prompt_cfg = cfg.data.prompt
    prompt_template = cfg.evaluation.prompt_template

    model.eval()

    predictions: List[str] = []
    references: List[str] = []
    records: List[Dict[str, Any]] = []

    for example in eval_examples:
        user_prompt = prompt_cfg["user_prompt_template"].format(
            tags=example[cfg.data.aspect_column]
        )
        prompt = prompt_template.format(
            system_prompt=prompt_cfg["system_prompt"],
            user_prompt=user_prompt,
        )
        generated = _generate_caption(
            model,
            tokenizer,
            prompt,
            cfg.evaluation.max_new_tokens,
            cfg.evaluation.temperature,
        )
        predictions.append(generated)
        references.append(example[cfg.data.caption_column])
        records.append(
            {
                "aspect_list": example[cfg.data.aspect_column],
                "reference": example[cfg.data.caption_column],
                "prediction": generated,
            }
        )

    metrics = _compute_metrics(predictions, references, cfg.evaluation.metrics)

    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics), indent=2)

    if cfg.evaluation.output_predictions:
        predictions_path = output_dir / cfg.evaluation.predictions_file
        pd.DataFrame(records).to_csv(predictions_path, index=False)

    return metrics


def caption_generation(cfg: CaptionGenerationConfig) -> None:
    log.info("Setting random seed...")
    set_seed(cfg.seed)

    logger = instantiate_loggers(cfg.get("logger"))
    wandb.login()
    if wandb.run is not None:
        wandb.define_metric("trainer/global_step", summary="max")
        wandb.define_metric("trainer/epoch", summary="max")
        wandb.define_metric("train/*", step="trainer/global_step")
        wandb.define_metric("eval/*", step="trainer/global_step")

    log.info("Preparing datasets...")
    dataset, test_examples = _prepare_datasets(cfg.data)
    log.info(
        f"Dataset prepared with {len(dataset['train'])} training and {len(dataset['validation'])} validation samples.",
    )

    log.info("Loading tokenizer...")
    tokenizer = _prepare_tokenizer(cfg.model)

    log.info("Loading model...")
    model, lora_config = _prepare_model(cfg)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Instantiating trainer...")
    trainer = _prepare_trainer(cfg, model, tokenizer, dataset, lora_config)

    log.info("Starting training...")
    trainer.train()

    log.info("Saving final model...")
    trainer.save_model(output_dir / "final_model")

    log.info("Running evaluation...")
    metrics = _run_evaluation(cfg, model, tokenizer, test_examples, output_dir)

    if metrics:
        log.info(f"Evaluation metrics: {metrics}")
        if wandb.run is not None:
            payload: Dict[str, float] = {}
            for key, value in metrics.items():
                payload.update(_flatten_numeric_metrics(value, f"eval/{key}"))
            payload["trainer/global_step"] = trainer.state.global_step
            if trainer.state.epoch is not None:
                payload["trainer/epoch"] = trainer.state.epoch
            wandb.log(payload, step=trainer.state.global_step)

    if metrics:
        log.info(f"Evaluation metrics: {metrics}")


@hydra.main(version_base=None, config_path="../../config", config_name="caption_generation")
def main(cfg: CaptionGenerationConfig) -> None:
    print_config_tree(cfg)
    caption_generation(cfg)


if __name__ == "__main__":
    main()
