from __future__ import annotations

import time
import torch
import hydra
import rootutils
import wandb
import lightning as pl

from transformers.trainer_utils import set_seed
from pathlib import Path

from src.caption import CaptionGenerationConfig, run_training, run_evaluation
from src.utils import print_config_tree, RankedLogger, instantiate_loggers
from src.caption.data import prepare_datasets
from src.caption.modeling import prepare_training_model, prepare_tokenizer
from src.caption.trainer import create_trainer
from src.caption.evaluation import MetricComputer
from datasets import load_dataset


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base=None, config_path="../../../config", config_name="caption_fine_tuning")
def main(cfg: CaptionGenerationConfig) -> None:
    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    logger = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

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
    trainer = create_trainer(cfg, model, tokenizer, dataset, lora_config, metric_computer, logger=logger)

    log.info("Starting training...")
    trainer.train()

    log.info("Saving final model...")
    trainer.save_model(model_dir / cfg.model.name / cfg.run_id)
    log.info(f"Saved model to {model_dir / cfg.model.name / cfg.run_id}")


if __name__ == "__main__":
    main()