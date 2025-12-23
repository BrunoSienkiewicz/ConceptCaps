"""Configuration dataclasses for VAE training."""
from dataclasses import dataclass, field
from typing import List, Optional, Union

from omegaconf import DictConfig


@dataclass
class VAEModelConfig:
    """Configuration for the VAE model architecture."""
    name: str = "MultiLabelVAE"
    input_dim: int = 0  # Will be computed based on taxonomy
    latent_dim: int = 32
    hidden_dim: int = 128
    dropout_p: float = 0.3
    beta: float = 1.0
    use_batch_norm: bool = False


@dataclass
class VAEDataConfig:
    """Configuration for data loading and preprocessing."""
    taxonomy_path: str = "data/concepts_to_tags.json"
    dataset_name: str = "google/MusicCaps"
    dataset_split: str = "train"
    aspect_column: str = "aspect_list"
    batch_size: int = 32
    dataloader_num_workers: int = 4
    shuffle: bool = True


@dataclass
class VAETrainerConfig:
    """Configuration for PyTorch Lightning trainer."""
    max_epochs: int = 250
    accelerator: str = "auto"
    devices: Union[str, int, List[int]] = "auto"
    strategy: Union[str, dict, None] = None
    precision: Union[str, int] = "32"
    gradient_clip_val: Optional[float] = 1.0
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
    """Configuration for loss function."""
    bce_weight: float = 1.0
    kld_weight: float = 1.0
    use_binary_cross_entropy: bool = True


@dataclass
class BetaVAEConfig:
    """Configuration for Beta-VAE variant."""
    beta: float = 4.0  # Weight for KL divergence term
    # Higher beta encourages disentanglement but may reduce reconstruction quality
    # Typical values: 1.0 (standard VAE), 4.0 (balanced), 10.0+ (high disentanglement)


@dataclass
class VAEPathsConfig:
    root_dir: str
    output_dir: str
    log_dir: str
    model_dir: str
    data_dir: str


@dataclass
class VAEInferenceConfig:
    """Configuration for VAE inference on sampled latent vectors."""
    enabled: bool = True
    num_samples: int = 1000
    temperature: float = 1.0
    threshold: float = 0.5
    seed: Optional[int] = None


@dataclass
class VAEConfig:
    """Main configuration for VAE training."""
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
