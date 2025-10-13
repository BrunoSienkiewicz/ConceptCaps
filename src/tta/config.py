from dataclasses import dataclass
from typing import Optional, Union

from omegaconf import DictConfig


@dataclass
class LoggerConfig:
    name: str
    version: str
    log_dir: str
    experiment_name: str

@dataclass
class PathsConfig:
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str

@dataclass
class DatasetConfig:
    name: str
    sample_size: int

@dataclass
class ModelConfig:
    name: str
    batch_size: int

@dataclass
class TTAConfig(DictConfig):
    model: ModelConfig
    dataset: DatasetConfig
    logger: LoggerConfig
    paths: PathsConfig
    random_state: int
    device: str = "cuda:0"
