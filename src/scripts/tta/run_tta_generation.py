from pathlib import Path

import wandb
import time
import hydra
import rootutils
import torch
import pytorch_lightning as pl

from src.utils import (RankedLogger, instantiate_loggers,
                       print_config_tree)
from src.tta.audio import generate_audio_samples
from src.tta.config import TTAConfig
from src.tta.data import prepare_dataloader, save_dataframe_metadata
from src.tta.modelling import prepare_model, prepare_tokenizer


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tta_generation")
def main(cfg: TTAConfig):
    log.info("Setting random seed...")
    pl.seed_everything(cfg.random_state)

    loggers = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    log.info("Preparing data...")
    processor = prepare_tokenizer(cfg.model)
    dataloader, df = prepare_dataloader(cfg.data, processor)
    dataloader.to(device)

    log.info("Loading model...")
    model = prepare_model(cfg.model)
    model.to(device)

    data_dir = Path(cfg.paths.data_dir) / cfg.model.name / cfg.run_id
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

    if cfg.evaluation.enabled:
        log.info("Starting evaluation...")
        # Call evaluation script
        from src.scripts.tta.run_evaluation import main as evaluation_main
        evaluation_main()

if __name__ == "__main__":
    main()

