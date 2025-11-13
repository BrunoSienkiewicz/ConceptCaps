from __future__ import annotations

import time
import torch
import hydra
import rootutils
import wandb

from transformers.trainer_utils import set_seed

from src.caption import CaptionGenerationConfig, run_training, run_evaluation
from src.utils import print_config_tree, RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="caption_fine_tuning")
def main(cfg: CaptionGenerationConfig) -> None:
    log = RankedLogger(__name__, rank_zero_only=True)

    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    _ = instantiate_loggers(cfg.get("logger"))
    wandb.login()

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    run_training(log, cfg)

    time.sleep(5)  # Ensure training deallocations are complete

    if (cfg.evaluation.enabled):
        run_evaluation(log, cfg)

if __name__ == "__main__":
    main()