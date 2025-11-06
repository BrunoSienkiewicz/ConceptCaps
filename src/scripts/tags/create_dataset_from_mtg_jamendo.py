from __future__ import annotations

import logging
import rootutils
import hydra
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from src.utils import print_config_tree, RankedLogger, instantiate_loggers
from src.tags.create_mtg_jamendo_tags_dataset import MTGJamendoDatasetCreator


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tags_dataset")
def create_dataset(cfg: DictConfig) -> Dict[str, Any]:
    """
    Create MTG Jamendo dataset using Hydra configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary with creation status and file paths
    """
    log = RankedLogger(__name__, rank_zero_only=True)

    log.info("=" * 80)
    log.info("MTG Jamendo Dataset Creation")
    log.info("=" * 80)
    log.info("\nConfiguration:")
    log.info(OmegaConf.to_yaml(cfg.data))

    # Create dataset
    creator = MTGJamendoDatasetCreator(
        output_dir=Path(cfg.data.output_dir),
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        random_state=cfg.data.random_state,
        logger=log,
    )

    creator.create_dataset(split="full")

    log.info("=" * 80)
    log.info("Dataset creation finished!")
    log.info("=" * 80)

    if cfg.data.push_to_hub:
        repo_name = cfg.data.hub_repo_name
        private = cfg.data.hub_private_repo
        log.info(f"Pushing dataset to HuggingFace Hub: {repo_name} (private={private})")
        creator.push_to_hub(repo_name, private=private)

    return {
        "status": "success",
        "output_dir": str(creator.output_dir),
        "train_file": str(creator.output_dir / "train.csv"),
        "validation_file": str(creator.output_dir / "validation.csv"),
        "test_file": str(creator.output_dir / "test.csv"),
    }


if __name__ == "__main__":
    create_dataset()
