"""Variational Autoencoder (VAE) module for music representation learning."""

from src.vae.config import (
    VAEConfig,
    VAEDataConfig,
    VAELossConfig,
    VAEModelConfig,
)
from src.vae.data import VAEDataModule
from src.vae.evaluation import MetricsSaveCallback, MetricsSaver, VAEMetrics
from src.vae.lightning_module import BetaVAELightningModule
from src.vae.model import BetaVAE

__all__ = [
    "BetaVAE",
    "BetaVAELightningModule",
    "MetricsSaveCallback",
    "MetricsSaver",
    "VAEConfig",
    "VAEDataConfig",
    "VAEDataModule",
    "VAELossConfig",
    "VAEMetrics",
    "VAEModelConfig",
]
