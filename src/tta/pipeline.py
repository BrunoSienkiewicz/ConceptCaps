from __future__ import annotations

from pathlib import Path

import rootutils
import hydra
import pytorch_lightning as pl
import torch
import wandb

from frechet_audio_distance import CLAPScore, FrechetAudioDistance

from src.tta.audio import generate_audio_samples
from src.tta.config import TTAConfig
from src.utils import RankedLogger, instantiate_loggers

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

def run_tta_generation(cfg: TTAConfig) -> None:
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
    
    for evaluation_name, evaluation_cfg in cfg.evaluation.items():
        log.info(f"Running evaluation: {evaluation_name}")
        if evaluation_cfg.type == "fad":
            fad_computer = FrechetAudioDistance(
                device=device,
                feature_extractor_name=evaluation_cfg.feature_extractor_name,
            )
            score = fad_computer.compute_fad(
                real_audio_dir=output_root / "real_audio",
                generated_audio_dir=output_root / "tta_generation",
            )
            log.info(f"FAD Score: {score}")
        elif evaluation_cfg.type == "clap":
            clap_computer = CLAPScore(
                device=device,
                model_name=evaluation_cfg.model_name,
            )
            score = clap_computer.compute_clap_score(
                real_audio_dir=output_root / "real_audio",
                generated_audio_dir=output_root / "tta_generation",
            )
            log.info(f"CLAP Score: {score}")
        else:
            log.warning(f"Unknown evaluation type: {evaluation_cfg.type}")

    if experiment_loggers:
        log.info("TTA evaluation completed and logged.")