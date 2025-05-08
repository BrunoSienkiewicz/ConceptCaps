from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class TCAVConfig:
    model_id: str
    data_path: str
    concept_name: str
    genre: str
    layers: list[str]
    random_state: int = 42
    batch_size: int = 1
    num_samples: int = 100
    experimental_set_size: int = 5
    n_groups: int = 10
    model_name: str = "facebook/musicgen-small"
    processor_name: str = "facebook/musicgen-small"
    output_path: str = "output"
