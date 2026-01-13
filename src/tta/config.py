from dataclasses import dataclass, field
from typing import List, Optional, Union

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
class DatasetConfig:
    dataset_name: str = ""
    id_column: str = "id"
    caption_column: str = "caption"
    remove_columns: Optional[List[str]] = None
    batch_size: int = 8

@dataclass
class PathsConfig:
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str

@dataclass
class ModelTokenizerConfig:
    padding_side: str = "right"
    use_fast: Optional[bool] = None
    pad_token_as_eos: bool = True
    max_new_tokens: int = 255

@dataclass
class ModelConfig:
    name: str
    checkpoint_dir: str = ""
    device_map: Union[str, dict, None] = "auto"
    trust_remote_code: bool = True
    tokenizer: ModelTokenizerConfig = field(default_factory=ModelTokenizerConfig)

@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    guidance_scale: Optional[float] = None

@dataclass
class EvaluationConfig:
    clap_model: str = "laion/clap-htsat-unfused"
    fad_model: str = "google/vggish"
    skip_evaluation: bool = False

@dataclass
class TTAConfig(DictConfig):
    model: ModelConfig
    data: DatasetConfig
    evaluation: EvaluationConfig
    generation: GenerationConfig
    logger: LoggerConfig
    paths: PathsConfig
    random_state: int
    batch_size: int = 8
    device: str = "cuda:0"
