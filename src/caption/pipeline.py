from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict

import torch
import wandb

from datasets import load_dataset

from src.caption.config import CaptionGenerationConfig
from src.caption.data import prepare_datasets, prepare_inference_datasets
from src.caption.evaluation import run_test_evaluation, MetricComputer
from src.caption.inference import run_caption_inference
from src.caption.modeling import prepare_training_model, prepare_tokenizer, prepare_evaluation_model_tokenizer
from src.caption.trainer import create_trainer
from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)

def run_training(cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device)

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
    model, lora_config = prepare_training_model(cfg.model, cfg.lora)
    model.to(device)

    metric_computer = MetricComputer(cfg.evaluation.metrics, tokenizer)

    log.info("Instantiating trainer...")
    trainer = create_trainer(cfg, model, tokenizer, dataset, lora_config, metric_computer)

    log.info("Starting training...")
    trainer.train()

    log.info("Saving final model...")
    trainer.save_model(model_dir / cfg.model.name / cfg.run_id)


def run_evaluation(cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    """
    Run evaluation on test set with batch processing.
    
    Args:
        log: Logger instance
        cfg: Configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    dataset = load_dataset(cfg.data.dataset_name)
    dataset = prepare_datasets(cfg.data, cfg.prompt, dataset)
    test_examples = dataset["test"]

    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    metric_computer = MetricComputer(cfg.evaluation.metrics, tokenizer)

    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running evaluation...")
    log.info(f"Test examples count: {len(test_examples)}")
    model, tokenizer = prepare_evaluation_model_tokenizer(log, cfg.model)

    model.to(device)
    
    output_dir = Path(cfg.paths.data_dir) / cfg.model.name / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_test_evaluation(
        cfg, 
        metric_computer, 
        model, 
        tokenizer, 
        test_examples, 
        output_dir,
        log,
    )

    if metrics and wandb.run is not None:
        payload: Dict[str, float] = {}
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue  # skip non-numeric values
            payload[f"test/{key}"] = value
        wandb.log(payload)

    return metrics


def run_inference(cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    """
    Run inference on test set with batch processing.
    
    Args:
        log: Logger instance
        cfg: Configuration
        
    Returns:
        DataFrame with predictions
    """
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    dataset = load_dataset(cfg.data.dataset_name)
    dataset = prepare_inference_datasets(cfg.data, cfg.prompt, dataset)

    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running inference...")
    model, tokenizer = prepare_evaluation_model_tokenizer(log, cfg.model)
    model.to(device)
    
    for split in dataset.keys():
        examples = dataset[split]
        predictions_path = data_dir / cfg.model.name / cfg.run_id / f"{split}_predictions.csv"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = run_caption_inference(
            cfg,
            model,
            tokenizer,
            examples,
            predictions_path,
            log,
        )
        log.info(f"Saved {len(results_df)} predictions for split '{split}' to: {predictions_path}")

    