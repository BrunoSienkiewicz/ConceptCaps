"""Configuration classes for Text-to-Audio generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from src.constants import (
    CLAP_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)


@dataclass
class LoggerConfig:
    name: str
    version: str
    log_dir: str
    experiment_name: str


@dataclass
class DatasetConfig:
    dataset_name: str = "default/dataset"
    id_column: str = "id"
    caption_column: str = "caption"
    remove_columns: Optional[List[str]] = None
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class PathsConfig:
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str

    def __post_init__(self) -> None:
        """Ensure all paths are Path objects."""
        self.output_dir = str(Path(self.output_dir).resolve())
        self.log_dir = str(Path(self.log_dir).resolve())
        self.model_dir = str(Path(self.model_dir).resolve())
        self.data_dir = str(Path(self.data_dir).resolve())


@dataclass
class ModelTokenizerConfig:
    padding_side: str = "right"
    use_fast: Optional[bool] = None
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


@dataclass
class ModelConfig:
    name: str
    checkpoint_dir: str = ""
    device_map: Union[str, Dict, None] = "auto"
    trust_remote_code: bool = True
    tokenizer: ModelTokenizerConfig = field(
        default_factory=ModelTokenizerConfig
    )


@dataclass
class GenerationConfig:
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = True
    guidance_scale: Optional[float] = DEFAULT_GUIDANCE_SCALE
    sample_rate: Optional[int] = None  # Will use model default if None
    use_accelerator: bool = False


@dataclass
class EvaluationConfig:
    clap_model: str = CLAP_MODEL
    fad_model: str = CLAP_MODEL
    skip_evaluation: bool = False
    compute_fad: bool = True
    reference_audio_dir: str = ""
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class TTAConfig(DictConfig):
    model: ModelConfig
    data: DatasetConfig
    evaluation: EvaluationConfig
    generation: GenerationConfig
    logger: LoggerConfig
    paths: PathsConfig
    random_state: int
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "cuda"
    run_id: str = "default"
