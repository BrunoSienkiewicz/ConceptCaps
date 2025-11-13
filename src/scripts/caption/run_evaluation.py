from __future__ import annotations

import torch
import hydra
import rootutils
import wandb

from transformers.trainer_utils import set_seed

from src.caption import CaptionGenerationConfig, run_evaluation
from src.utils import print_config_tree, RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base=None, config_path="../../../config", config_name="caption_evaluation")
def main(cfg: CaptionGenerationConfig) -> None:
    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    loggers = instantiate_loggers(cfg.get("logger"))
    for logger in loggers:
        if isinstance(logger, wandb.WandbLogger):
            wandb.login()
            run_name = logger.run_name if logger.run_name else None
            if run_name is None:
                run_name = f"caption-evaluation-{cfg.model.name}-{int(torch.randint(0, 1e6, (1,)).item())}"
            wandb.init(project=logger.project, name=run_name)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    run_evaluation(cfg)

if __name__ == "__main__":
    main()