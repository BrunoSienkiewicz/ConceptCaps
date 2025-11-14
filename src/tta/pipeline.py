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
from src.tta.data import prepare_dataloader
from src.tta.modelling import prepare_model, prepare_tokenizer
from src.utils import RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

def run_tta(cfg: TTAConfig) -> None:
    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    _ = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info("Preparing data...")
    processor = prepare_tokenizer(cfg.model)
    dataloader, df = prepare_dataloader(cfg.data, processor)
    dataloader.to(device)

    log.info("Loading model...")
    model = prepare_model(cfg.model)
    model.to(device)

    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating audio samples...")
    generate_audio_samples(
        model,
        dataloader,
        data_dir,
        cfg.model.model.max_new_tokens,
        cfg.data.batch_size,
    )

    # Add path to generated samples to dataframe
    df['audio_path'] = df.index.apply(
        lambda x: str(data_dir / f"{x}.wav")
    )
    df.to_csv(data_dir / "metadata.csv", index=False)

    log.info("TTA generation completed and logged.")
    

def evaluate_tta(cfg: TTAConfig) -> None:
    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    _ = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

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
            if wandb.run is not None:
                wandb.log({f"FAD/{evaluation_name}": score})
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
            if wandb.run is not None:
                wandb.log({f"CLAP/{evaluation_name}": score})
        else:
            log.warning(f"Unknown evaluation type: {evaluation_cfg.type}")

    log.info("TTA evaluation completed and logged.")