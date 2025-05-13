from dataclasses import dataclass

from music_gen_interpretability.data.generic_data_module import GenericConceptDataModule
from music_gen_interpretability.tcav.model import ConceptClassifier
from transformers import MusicgenForConditionalGeneration, AutoProcessor


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
    concept_name: str
    genre: str
    layers: list[str]
    layer_attr_method: str
    n_groups: int
    experimental_set_size: int

@dataclass
class TCAVConfig:
    model: ModelConfig
    experiment: ExperimentConfig
    data: GenericConceptDataModule
