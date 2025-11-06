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


@dataclass
class ModelConfig:
    name: str
    checkpoint_dir: str = ""
    device_map: Union[str, dict, None] = "auto"
    trust_remote_code: bool = True
    quantization: Optional[QuantizationConfig] = field(default_factory=QuantizationConfig)
    tokenizer: ModelTokenizerConfig = field(default_factory=ModelTokenizerConfig)


@dataclass
class PromptTemplateConfig:
    template: str
    system_prompt: str
    user_prompt_template: str


@dataclass
class DatasetConfig:
    create: bool = False
    dataset_name: str = ""
    train_file: str = ""
    validation_file: str = ""
    test_file: str = ""
    push_to_hub: bool = False
    hub_repo_name: str = ""
    hub_private_repo: bool = True
    aspect_column: str = "aspect_list"
    caption_column: str = "caption"
    text_column: str = "text"
    prompt: Optional[PromptTemplateConfig] = field(default=None)
    remove_columns: Optional[List[str]] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


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
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: List[str] = field(default_factory=lambda: ["none"])
    logging_steps: int = 1
    logging_strategy: str = "steps"
    save_strategy: str = "no"
    load_best_model_at_end: bool = True
    save_only_model: bool = False
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None


@dataclass
class EvaluationMetricConfig:
    name: str
    kwargs: dict = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    enabled: bool = True
    max_new_tokens: int = 256
    metrics: List[EvaluationMetricConfig] = field(default_factory=list)
    output_predictions: bool = True
    predictions_file: str = "predictions.csv"


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
    lora: LoRAConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    device: str = "cuda"
    random_state: int = 42
