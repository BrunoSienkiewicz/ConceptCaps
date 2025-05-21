from dataclasses import dataclass

from transformers import AutoProcessor, MusicgenForConditionalGeneration

from music_gen_interpretability.data.generic_data_module import GenericDataModule
from music_gen_interpretability.tcav.model import ConceptClassifier


@dataclass
class ModelConfig:
    model_id: str
    processor: AutoProcessor
    model: MusicgenForConditionalGeneration
    model_name: str
    processor_name: str
    classifier: ConceptClassifier


@dataclass
class ExperimentConfig:
    random_state: int
    layers: list[str]
    layer_attr_method: str
    n_groups: int
    experimental_set_size: int
    num_samples: int
    output_dir: str


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
class TCAVConfig:
    model: ModelConfig
    experiment: ExperimentConfig
    data: GenericDataModule
    logger: LoggerConfig
    paths: PathsConfig
