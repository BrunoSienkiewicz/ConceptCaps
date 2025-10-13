from pathlib import Path

import hydra
import rootutils
import pytorch_lightning as pl

from src.utils import (RankedLogger, instantiate_loggers, log_hyperparameters,
                       print_config_tree)
from src.tta.config import TTAConfig
from src.tta.helper_methods import preprocess_dataset, sample_dataset, generate_audio

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)

def tta(cfg: TTAConfig):
    random_state = cfg.random_state
    pl.seed_everything(random_state)

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    inputs = sample_dataset(cfg.dataset)

    for batch in split(inputs, cfg.model.batch_size):
        tensors = preprocess_dataset(batch)

        audio = generate_audio(cfg.model, tensors)


@hydra.main(version_base=None, config_path="../../config", config_name="tcav")
def main(cfg: TTAConfig):
    print_config_tree(cfg)
    tta(cfg)


if __name__ == "__main__":
    main()
