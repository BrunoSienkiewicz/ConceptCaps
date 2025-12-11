from dataclasses import dataclass, field
from typing import List, Optional, Union

from omegaconf import DictConfig


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
    pad_token: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    checkpoint_dir: str = ""
    device_map: Union[str, dict, None] = "auto"
    trust_remote_code: bool = True
    quantization: Optional[QuantizationConfig] = field(default_factory=QuantizationConfig)
    tokenizer: ModelTokenizerConfig = field(default_factory=ModelTokenizerConfig)


@dataclass
class PromptConfig:
    template: str
    system_prompt: str
    user_prompt_template: str
    pad_token: Optional[str] = None

@dataclass
class DatasetConfig:
    dataset_name: str = ""
    aspect_column: str = "aspect_list"
    caption_column: str = "caption"
    text_column: str = "text"
    id_column: str = "id"
    batch_size: int = 8
    dataloader_num_workers: int = 4
    remove_columns: Optional[List[str]] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=list)
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainerConfig:
    max_epochs: int = 3
    dataloader_num_workers: int = 1
    accelerator: str = "auto"
    devices: Union[str, int, List[int]] = "auto"
    strategy: Union[str, dict, None] = "ddp"
    precision: Union[str, int] = "bf16"
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 5
    val_check_interval: Union[float, int, None] = 1.0
    check_val_every_n_epoch: int = 1
    enable_progress_bar: bool = False
    enable_model_summary: bool = False
    deterministic: bool = False
    optimizer: DictConfig = field(default_factory=lambda: DictConfig({}))
    lr_scheduler: DictConfig = field(default_factory=lambda: DictConfig({}))


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    prompt: PromptConfig = field(default_factory=PromptConfig)


@dataclass
class EvaluationMetricConfig:
    name: str
    kwargs: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    metrics: List[EvaluationMetricConfig] = field(default_factory=list)
    output_predictions: bool = True


@dataclass
class PathsConfig:
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str


@dataclass
class CaptionGenerationConfig(DictConfig):
    model: ModelConfig
    data: DatasetConfig
    prompt: PromptConfig
    lora: LoRAConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    generation: GenerationConfig
    device: str = "cuda"
    random_state: int = 42
    run_id: str = "default_run"
    task_name: str = "caption_generation"
