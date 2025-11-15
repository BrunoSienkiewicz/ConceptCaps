from __future__ import annotations

from pathlib import Path

import rootutils
import hydra
import pytorch_lightning as pl
import torch
import wandb

from src.tta.audio import generate_audio_samples
from src.tta.config import TTAConfig
from src.tta.data import prepare_dataloader, save_dataframe_metadata
from src.tta.modelling import prepare_model, prepare_tokenizer
from src.utils import RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

def run_tta(cfg: TTAConfig) -> None:
    pl.seed_everything(cfg.random_state)
    device = torch.device(cfg.device)

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
        data_dir / "audio_samples",
        cfg.model.model.max_new_tokens,
        cfg.data.batch_size,
        df,
        id_column=cfg.data.get("id_column", "id"),
        filename_template=cfg.data.get("filename_template", "{}.wav"),
    )

    log.info("Saving metadata...")
    save_dataframe_metadata(
        df,
        data_dir,
        id_column=cfg.data.get("id_column", "id"),
        filename_template=cfg.data.get("filename_template", "{}.wav"),
    )

    log.info("TTA generation completed and logged.")
    

def evaluate_tta(cfg: TTAConfig) -> None:
    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    _ = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info("Evaluating TTA generated samples...")
    

    log.info("TTA evaluation completed and logged.")