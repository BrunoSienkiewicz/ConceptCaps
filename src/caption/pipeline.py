from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gc
import torch
import wandb

from peft import PeftModel
from transformers.trainer_utils import set_seed

from src.caption.config import CaptionGenerationConfig
from src.caption.data import prepare_datasets
from src.caption.evaluation import run_evaluation
from src.caption.logging_utils import flatten_numeric_metrics
from src.caption.modeling import prepare_model, prepare_tokenizer, prepare_evaluation_model_tokenizer
from src.caption.trainer import create_trainer
from src.utils import RankedLogger, instantiate_loggers


def run_caption_generation(cfg: CaptionGenerationConfig) -> Dict[str, Any]:
    log = RankedLogger(__name__, rank_zero_only=True)

    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    _ = instantiate_loggers(cfg.get("logger"))
    wandb.login()

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info("Preparing datasets...")
    dataset, test_examples = prepare_datasets(cfg.data)
    log.info(
        f"Dataset prepared with {len(dataset['train'])} training and {len(dataset['validation'])} validation samples.",
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

    log.info("Instantiating trainer...")
    trainer = create_trainer(cfg, model, tokenizer, dataset, lora_config)

    log.info("Starting training...")
    # trainer.train()

    log.info("Saving final model...")
    trainer.save_model(model_dir / "final_model")

    # Cleanup for evaluation
    del trainer
    del model
    torch.cuda.empty_cache()
    gc.collect()

    log.info("Running evaluation...")
    log.info(f"Test examples count: {len(test_examples)}")
    eval_model, eval_tokenizer = prepare_evaluation_model_tokenizer(cfg.model)

    eval_model = PeftModel.from_pretrained(
        eval_model, 
        model_dir / "final_model",
        is_trainable=False
    )
    eval_model.to(device)
    metrics = run_evaluation(cfg, eval_model, eval_tokenizer, test_examples, output_dir, log)

    if metrics and wandb.run is not None:
        payload: Dict[str, float] = {}
        for key, value in metrics.items():
            payload.update(flatten_numeric_metrics(value, f"test/{key}"))
        wandb.log(payload)

    return metrics
