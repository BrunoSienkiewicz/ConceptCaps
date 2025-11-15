from __future__ import annotations

import time
import torch
import hydra
import rootutils
import wandb
import lightning as pl

from transformers.trainer_utils import set_seed

from src.caption import CaptionGenerationConfig, run_training, run_evaluation
from src.utils import print_config_tree, RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base=None, config_path="../../../config", config_name="caption_fine_tuning")
def main(cfg: CaptionGenerationConfig) -> None:
    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    run_training(cfg)


if __name__ == "__main__":
    main()