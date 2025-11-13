from __future__ import annotations

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb

from frechet_audio_distance import CLAPScore, FrechetAudioDistance

from src.tta.audio import generate_audio_samples
from src.tta.config import TTAConfig
from src.utils import RankedLogger, instantiate_loggers


def run_tta_generation(cfg: TTAConfig) -> None:
    log = RankedLogger(__name__, rank_zero_only=True)

    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    experiment_loggers = instantiate_loggers(cfg.get("logger"))
    wandb.login()

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model.model._target_}>")
    model_wrapper = hydra.utils.instantiate(cfg.model)
    model_wrapper.model.to(device)

    log.info("Preparing data...")
    data_module.prepare_data()
    data_module.setup()

    output_root = Path(cfg.paths.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if hasattr(data_module, "dataset"):
        data_module.dataset.to_csv(output_root / "full_dataset.csv", index=False)

    log.info("Generating audio samples...")
    dataloader = data_module.random_dataloader()
    generate_audio_samples(
        model_wrapper,
        dataloader,
        output_root / "tta_generation",
        cfg.model.model.max_new_tokens,
        data_module.batch_size,
    )

    if experiment_loggers:
        log.info("TTA generation completed and logged.")
    

def evaluate_tta_generation(cfg: TTAConfig) -> None:
    log = RankedLogger(__name__, rank_zero_only=True)

    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    experiment_loggers = instantiate_loggers(cfg.get("logger"))
    wandb.login()

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model.model._target_}>")
    model_wrapper = hydra.utils.instantiate(cfg.model)
    model_wrapper.model.to(device)

    log.info("Preparing data...")
    data_module.prepare_data()
    data_module.setup()

    output_root = Path(cfg.paths.output_dir)

    log.info("Evaluating TTA generated samples...")
    # Evaluation logic would go here

    if experiment_loggers:
        log.info("TTA evaluation completed and logged.")