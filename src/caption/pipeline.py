from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gc
import torch
import wandb

from peft import PeftModel
from datasets import load_dataset

from src.caption.config import CaptionGenerationConfig
from src.caption.data import prepare_datasets, create_datasets
from src.caption.evaluation import run_test_evaluation, MetricComputer
from src.caption.logging_utils import flatten_numeric_metrics
from src.caption.modeling import prepare_model, prepare_tokenizer, prepare_evaluation_model_tokenizer
from src.caption.trainer import create_trainer
from src.utils import RankedLogger, instantiate_loggers


def create_caption_generation_datasets(log, cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preparing datasets...")
    create_datasets(log, cfg.data, data_dir)

    if cfg.data.push_to_hub:
        data_files = {
            "train": cfg.data.train_file,
            "validation": cfg.data.validation_file,
            "test": cfg.data.test_file,
        }
        dataset = load_dataset("csv", data_files=data_files)
        dataset.push_to_hub(cfg.data.hub_repo_name, private=cfg.data.hub_private_repo)


def run_training(log, cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    log.info("Loading datasets...")
    data_files = {
        "train": cfg.data.train_file,
        "validation": cfg.data.validation_file,
        "test": cfg.data.test_file,
    }
    dataset = load_dataset("csv", data_files=data_files)
    # dataset = load_dataset(cfg.data.hub_repo_name)
    dataset = prepare_datasets(cfg.data, dataset)
    log.info(
        f"Dataset loaded with {len(dataset['train'])} training and {len(dataset['validation'])} validation samples.",
    )

    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    log.info("Loading model...")
    model, lora_config = prepare_model(cfg.model, cfg.lora)
    model.to(device)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True) 

    metric_computer = MetricComputer(cfg.evaluation.metrics)

    log.info("Instantiating trainer...")
    trainer = create_trainer(cfg, model, tokenizer, dataset, lora_config, metric_computer)

    log.info("Starting training...")
    trainer.train()

    log.info("Saving final model...")
    trainer.save_model(model_dir / "final_model")


def run_evaluation(log, cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    dataset = load_dataset(cfg.data.hub_repo_name)
    dataset = prepare_datasets(cfg.data, dataset)
    test_examples = dataset["test"]

    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    metric_computer = MetricComputer(cfg.evaluation.metrics, tokenizer)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True) 

    log.info("Running evaluation...")
    log.info(f"Test examples count: {len(test_examples)}")
    eval_model, eval_tokenizer = prepare_evaluation_model_tokenizer(cfg.model)

    eval_model = PeftModel.from_pretrained(
        eval_model, 
        model_dir / "final_model",
        is_trainable=False
    )
    eval_model.to(device)
    metrics = run_test_evaluation(cfg, metric_computer, eval_model, eval_tokenizer, test_examples, output_dir, log)

    if metrics and wandb.run is not None:
        payload: Dict[str, float] = {}
        for key, value in metrics.items():
            payload.update(flatten_numeric_metrics(value, f"test/{key}"))
        wandb.log(payload)

    return metrics
