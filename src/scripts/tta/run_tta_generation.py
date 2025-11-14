from pathlib import Path

import wandb
import time
import hydra
import rootutils
import torch
import pytorch_lightning as pl

from transformers.trainer_utils import set_seed

from src.utils import (RankedLogger, instantiate_loggers, log_hyperparameters,
                       print_config_tree)
from src.tta import TTAConfig, run_tta


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tta_generation")
def main(cfg: TTAConfig):
    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    loggers = instantiate_loggers(cfg.get("logger"))
    for logger in loggers:
        if isinstance(logger, wandb.WandbLogger):
            wandb.login()
            run_name = logger.run_name if logger.run_name else None
            if run_name is None:
                run_name = f"tta-generation-{cfg.model.name}-{time.strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=logger.project, name=run_name)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)
    run_tta(cfg)


if __name__ == "__main__":
    main()

