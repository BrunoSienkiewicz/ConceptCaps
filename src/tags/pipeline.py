from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import gc
import torch
import wandb

from peft import PeftModel
from datasets import load_dataset

from src.tags.config import TagsConfig
from src.tags.data import create_datasets


def create_tags_datasets(log, cfg: TagsConfig) -> Dict[str, Any]:
    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preparing datasets...")
    create_datasets(log, cfg.data, data_dir)

    if cfg.data.push_to_hub:
        data_files = {
            "train": cfg.data.train_file,
            "validation": cfg.data.validation_file,
            "test": cfg.data.test_file,
        }
        dataset = load_dataset("csv", data_files=data_files)
        dataset.push_to_hub(cfg.data.hub_repo_name, private=cfg.data.hub_private_repo)

        
