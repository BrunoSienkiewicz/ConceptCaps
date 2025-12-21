"""Lightning module for VAE training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from typing import Optional, Any, Union, Dict

from src.vae.modeling import MultiLabelVAE, BetaVAE
from src.vae.config import VAEModelConfig, VAELossConfig
from src.vae.metrics import VAEMetrics


class VAELightningModule(pl.LightningModule):
    """Lightning module for training VAE on multi-label tag data."""
    
    def __init__(
        self,
        model_cfg: VAEModelConfig,
        loss_cfg: VAELossConfig,
        learning_rate: float = 5e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        
        # Initialize VAE model
        self.model = MultiLabelVAE(
            input_dim=model_cfg.input_dim,
            latent_dim=model_cfg.latent_dim,
            hidden_dim=model_cfg.hidden_dim,
            dropout_p=model_cfg.dropout_p,
            use_batch_norm=model_cfg.use_batch_norm,
        )
        
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through VAE."""
        return self.model(x)
    
    def _compute_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        """Compute VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss (Binary Cross Entropy)
        if self.loss_cfg.use_binary_cross_entropy:
            bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            bce = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Weighted combination
        total_loss = (
            self.loss_cfg.bce_weight * bce +
            self.loss_cfg.kld_weight * kld
        )
        
        return {
            "loss": total_loss,
            "bce": bce,
            "kld": kld,
        }
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch[0]
        recon, mu, logvar = self(x)
        
        loss_dict = self._compute_loss(recon, x, mu, logvar)
        
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce", loss_dict["bce"], on_step=False, on_epoch=True)
        self.log("train/kld", loss_dict["kld"], on_step=False, on_epoch=True)
        
        return loss_dict["loss"]
    
    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """Validation step."""
        x = batch[0]
        recon, mu, logvar = self(x)
        
        loss_dict = self._compute_loss(recon, x, mu, logvar)
        
        self.log("val/loss", loss_dict["loss"], on_epoch=True, prog_bar=True)
        self.log("val/bce", loss_dict["bce"], on_epoch=True)
        self.log("val/kld", loss_dict["kld"], on_epoch=True)
        
        # Compute evaluation metrics
        metrics = VAEMetrics.compute_all_metrics(recon, x, mu, threshold=0.5)
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_epoch=True)
        
        return {**loss_dict, **metrics}
    
    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        """Test step."""
        x = batch[0]
        recon, mu, logvar = self(x)
        
        loss_dict = self._compute_loss(recon, x, mu, logvar)
        
        self.log("test/loss", loss_dict["loss"], on_epoch=True, prog_bar=True)
        self.log("test/bce", loss_dict["bce"], on_epoch=True)
        self.log("test/kld", loss_dict["kld"], on_epoch=True)
        
        # Compute evaluation metrics
        metrics = VAEMetrics.compute_all_metrics(recon, x, mu, threshold=0.5)
        for metric_name, metric_value in metrics.items():
            self.log(f"test/{metric_name}", metric_value, on_epoch=True)
        
        # Compute diversity metrics
        diversity_metrics = VAEMetrics.compute_diversity_metrics(recon, x, threshold=0.5)
        for metric_name, metric_value in diversity_metrics.items():
            self.log(f"test/{metric_name}", metric_value, on_epoch=True)
        
        return {**loss_dict, **metrics, **diversity_metrics}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return optimizer
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (mu only)."""
        mu, _ = self.model.encode(x)
        return mu
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the prior."""
        return self.model.sample(num_samples, device=self.device)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        return self.model.reconstruct(x)


class BetaVAELightningModule(pl.LightningModule):
    """Lightning module for training Beta-VAE on multi-label tag data.
    
    Beta-VAE uses a weighted KL divergence term to encourage disentangled
    representations of latent factors.
    """
    
    def __init__(
        self,
        model_cfg: VAEModelConfig,
        loss_cfg: VAELossConfig,
        learning_rate: float = 5e-4,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        beta: float = 4.0,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.loss_cfg = loss_cfg
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.beta = beta
        
        # Initialize Beta-VAE model
        self.model = BetaVAE(
            input_dim=model_cfg.input_dim,
            latent_dim=model_cfg.latent_dim,
            hidden_dim=model_cfg.hidden_dim,
            dropout_p=model_cfg.dropout_p,
            use_batch_norm=model_cfg.use_batch_norm,
            beta=beta,
        )
        
        self.save_hyperparameters()
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through Beta-VAE."""
        return self.model(x)
    
    def _compute_loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict:
        """Compute Beta-VAE loss with weighted KL divergence."""
        # Reconstruction loss (Binary Cross Entropy)
        if self.loss_cfg.use_binary_cross_entropy:
            bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            bce = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Beta-VAE: weight the KL term by beta for disentanglement
        # Higher beta encourages more disentangled representations
        total_loss = (
            self.loss_cfg.bce_weight * bce +
            self.loss_cfg.kld_weight * self.beta * kld
        )
        
        return {
            "loss": total_loss,
            "bce": bce,
            "kld": kld,
        }
    
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch[0]
        recon, mu, logvar = self(x)
        
        loss_dict = self._compute_loss(recon, x, mu, logvar)
        
        self.log("train/loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/bce", loss_dict["bce"], on_step=False, on_epoch=True)
        self.log("train/kld", loss_dict["kld"], on_step=False, on_epoch=True)
        
        return loss_dict["loss"]
    
    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """Validation step."""
        x = batch[0]
        recon, mu, logvar = self(x)
        
        loss_dict = self._compute_loss(recon, x, mu, logvar)
        
        self.log("val/loss", loss_dict["loss"], on_epoch=True, prog_bar=True)
        self.log("val/bce", loss_dict["bce"], on_epoch=True)
        self.log("val/kld", loss_dict["kld"], on_epoch=True)
        
        # Compute evaluation metrics
        metrics = VAEMetrics.compute_all_metrics(recon, x, mu, threshold=0.5)
        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, on_epoch=True)
        
        return {**loss_dict, **metrics}
    
    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        """Test step with diversity metrics."""
        x = batch[0]
        recon, mu, logvar = self(x)
        
        loss_dict = self._compute_loss(recon, x, mu, logvar)
        
        self.log("test/loss", loss_dict["loss"], on_epoch=True, prog_bar=True)
        self.log("test/bce", loss_dict["bce"], on_epoch=True)
        self.log("test/kld", loss_dict["kld"], on_epoch=True)
        
        # Compute evaluation metrics
        metrics = VAEMetrics.compute_all_metrics(recon, x, mu, threshold=0.5)
        for metric_name, metric_value in metrics.items():
            self.log(f"test/{metric_name}", metric_value, on_epoch=True)
        
        # Compute diversity metrics
        diversity_metrics = VAEMetrics.compute_diversity_metrics(recon, x, threshold=0.5)
        for metric_name, metric_value in diversity_metrics.items():
            self.log(f"test/{metric_name}", metric_value, on_epoch=True)
        
        return {**loss_dict, **metrics, **diversity_metrics}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return optimizer
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (mu only)."""
        mu, _ = self.model.encode(x)
        return mu
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the prior."""
        return self.model.sample(num_samples, device=self.device)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input."""
        return self.model.reconstruct(x)
