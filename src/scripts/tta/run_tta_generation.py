from pathlib import Path

import hydra
import rootutils
import torch
import pytorch_lightning as pl

from src.utils import (RankedLogger, instantiate_loggers, log_hyperparameters,
                       print_config_tree)
from src.tta import TTAConfig, run_tta


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../config", config_name="tta")
def main(cfg: TTAConfig):
    print_config_tree(cfg)
    run_tta(cfg)


if __name__ == "__main__":
    main()

