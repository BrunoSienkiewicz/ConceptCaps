from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gc
import torch
import wandb

from peft import PeftModel
from datasets import load_dataset

from src.caption.config import CaptionGenerationConfig
from src.caption.data import prepare_datasets, prepare_inference_datasets
from src.caption.evaluation import run_test_evaluation, MetricComputer
from src.caption.inference import run_inference
from src.caption.logging_utils import flatten_numeric_metrics
from src.caption.modeling import prepare_training_model, prepare_tokenizer, prepare_evaluation_model_tokenizer
from src.caption.trainer import create_trainer


def run_training(log, cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True) 

    log.info("Loading datasets...")
    dataset = load_dataset(cfg.data.dataset_name)
    dataset = prepare_datasets(cfg.data, cfg.prompt, dataset)
    log.info(
        f"Dataset loaded with {len(dataset['train'])} training and {len(dataset['validation'])} validation samples.",
    )

    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    log.info("Loading model...")
    model, lora_config = prepare_training_model(log, cfg.model, cfg.lora)
    model.to(device)

    metric_computer = MetricComputer(cfg.evaluation.metrics, tokenizer)

    log.info("Instantiating trainer...")
    trainer = create_trainer(cfg, model, tokenizer, dataset, lora_config, metric_computer)

    log.info("Starting training...")
    trainer.train()

    log.info("Saving final model...")
    trainer.save_model(model_dir)

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_evaluation(log, cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    dataset = load_dataset(cfg.data.dataset_name)
    dataset = prepare_datasets(cfg.data, cfg.prompt, dataset)
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
    model, tokenizer = prepare_evaluation_model_tokenizer(log, cfg.model)

    model.to(device)
    metrics = run_test_evaluation(cfg, metric_computer, model, tokenizer, test_examples, output_dir, log)

    if metrics and wandb.run is not None:
        payload: Dict[str, float] = {}
        for key, value in metrics.items():
            payload.update(flatten_numeric_metrics(value, f"test/{key}"))
        wandb.log(payload)

    return metrics


def run_inference(log, cfg: CaptionGenerationConfig):
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    dataset = load_dataset(cfg.data.dataset_name)
    dataset = prepare_inference_datasets(cfg.data, cfg.prompt, dataset)
    examples = dataset["all"]

    log.info("Loading tokenizer...")

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running inference...")
    model, tokenizer = prepare_evaluation_model_tokenizer(log, cfg.model)
    model.to(device)

    run_inference(cfg, model, tokenizer, examples, output_dir, log)

    