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


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class ModelTokenizerConfig:
    padding_side: str = "right"
    use_fast: Optional[bool] = None
    pad_token_as_eos: bool = True


@dataclass
class ModelConfig:
    name: str
    checkpoint_dir: str = ""
    device_map: Union[str, dict, None] = "auto"
    trust_remote_code: bool = True
    quantization: Optional[QuantizationConfig] = field(default_factory=QuantizationConfig)
    tokenizer: ModelTokenizerConfig = field(default_factory=ModelTokenizerConfig)

@dataclass
class PathsConfig:
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str


@dataclass
class PromptTemplateConfig:
    template: str
    system_prompt: str
    user_prompt_template: str


@dataclass
class TagsConfig:
    data: DatasetConfig
    model: ModelConfig
    prompt: PromptTemplateConfig
    paths: PathsConfig