"""Configuration dataclasses for VAE training and inference."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from src.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_CLIP_VAL,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
)


@dataclass
class VAEModelConfig:
    name: str = "MultiLabelVAE"
    input_dim: int = 0  # Will be computed based on taxonomy
    latent_dim: int = DEFAULT_LATENT_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    dropout_p: float = 0.3
    beta: float = 1.0
    use_batch_norm: bool = False


@dataclass
class VAEDataConfig:
    taxonomy_path: str = "data/concepts_to_tags.json"
    dataset_name: str = "google/MusicCaps"
    dataset_split: str = "train"
    aspect_column: str = "aspect_list"
    batch_size: int = DEFAULT_BATCH_SIZE
    dataloader_num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        """Validate and resolve paths."""
        self.taxonomy_path = str(Path(self.taxonomy_path).resolve())


@dataclass
class VAETrainerConfig:
    max_epochs: int = DEFAULT_MAX_EPOCHS
    accelerator: str = "auto"
    devices: Union[str, int, List[int]] = "auto"
    strategy: Union[str, Dict, None] = None
    precision: Union[str, int] = "32"
    gradient_clip_val: Optional[float] = DEFAULT_GRADIENT_CLIP_VAL
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 10
    val_check_interval: Union[float, int, None] = None
    check_val_every_n_epoch: Optional[int] = 1
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    deterministic: bool = False
    optimizer: DictConfig = field(default_factory=lambda: DictConfig({}))
    lr_scheduler: DictConfig = field(default_factory=lambda: DictConfig({}))


@dataclass
class VAELossConfig:
    bce_weight: float = 1.0
    kld_weight: float = 1.0
    use_binary_cross_entropy: bool = True


@dataclass
class VAEPathsConfig:
    root_dir: str
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str

    def __post_init__(self) -> None:
        """Ensure all paths are resolved."""
        self.root_dir = str(Path(self.root_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())
        self.log_dir = str(Path(self.log_dir).resolve())
        self.model_dir = str(Path(self.model_dir).resolve())
        self.data_dir = str(Path(self.data_dir).resolve())


@dataclass
class VAEInferenceConfig:
    enabled: bool = True
    num_samples: int = 1000
    temperature: float = 1.0
    threshold: float = 0.5
    seed: Optional[int] = None


@dataclass
class VAEConfig:
    model: VAEModelConfig = field(default_factory=VAEModelConfig)
    data: VAEDataConfig = field(default_factory=VAEDataConfig)
    trainer: VAETrainerConfig = field(default_factory=VAETrainerConfig)
    loss: VAELossConfig = field(default_factory=VAELossConfig)
    paths: VAEPathsConfig = field(default_factory=VAEPathsConfig)
    inference: VAEInferenceConfig = field(default_factory=VAEInferenceConfig)

    random_state: int = 42
    device: str = "cuda"
    run_id: str = "${now:%Y-%m-%d_%H-%M-%S}"
    task_name: str = "vae_training"
    model_name: str = "vae_final"
    save_model: bool = True
