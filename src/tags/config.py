from dataclasses import dataclass, field
from typing import List, Optional, Union

from omegaconf import DictConfig

@dataclass
class DatasetConfig:
    output_dir: str = "${paths.data_dir}/tags/"
    train_file: str = ""
    validation_file: str = ""
    test_file: str = ""
    train_split: float = 0.8
    val_split: float = 0.1
    push_to_hub: bool = False
    hub_repo_name: str = ""
    hub_private_repo: bool = True
    random_state: int = 42
