from dataclasses import dataclass
from typing import Optional, Union

from transformers import AutoProcessor, MusicgenForConditionalGeneration
from src.data.generic_data_module import GenericDataModule

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
class ModelConfig:
    model_id: str
    model: MusicgenForConditionalGeneration
    max_new_tokens: int = 255

@dataclass
class TTAConfig(DictConfig):
    model: ModelConfig
    data: GenericDataModule
    logger: LoggerConfig
    paths: PathsConfig
    random_state: int
    device: str = "cuda:0"
