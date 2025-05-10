from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

from music_gen_interpretability.data.generic_data_module import GenericDataModule
from music_gen_interpretability.tcav.model import ConceptClassifier
from transformers import AutoProcessor, MusicGenForConditionalGeneration

@dataclass
class ModelConfig:
    model_id: str
    processor: AutoProcessor
    model: MusicGenForConditionalGeneration
    model_name: str
    processor_name: str
    layer_attr_method: str
    classifier: ConceptClassifier

@dataclass
class ExperimentConfig:
    random_state: int
    concept_name: str
    genre: str
    layers: list[str]
    layer_attr_method: str
    n_groups: int

@dataclass
class TCAVConfig:
    model: ModelConfig
    experiment: ExperimentConfig
    data: GenericDataModule
