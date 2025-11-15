from __future__ import annotations
import time

import torch
import hydra
import rootutils
import wandb
import lightning as pl

from transformers.trainer_utils import set_seed

from src.caption import CaptionGenerationConfig, run_inference
from src.utils import print_config_tree, RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base=None, config_path="../../../config", config_name="caption_inference")
def main(cfg: CaptionGenerationConfig) -> None:
    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    loggers = instantiate_loggers(cfg.get("logger"))
    for logger in loggers:
        if isinstance(logger, pl.loggers.wandb.WandbLogger):
            wandb.login()
            run_name = logger.run_name if logger.run_name else None
            if run_name is None:
                run_name = f"caption-inference-{cfg.model.name}-{time.strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=logger.project, name=run_name)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    run_inference(cfg)

if __name__ == "__main__":
    main()