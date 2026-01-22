"""VAE model implementations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    """Beta-VAE: Disentangled Variational Autoencoder with weighted KL divergence.

    Reference: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a
    Constrained Variational Framework"
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout_p: float = 0.3,
        use_batch_norm: bool = False,
        beta: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn1 = None

        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        if use_batch_norm:
            self.bn2 = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn2 = None

        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Dropout for denoising
        self.input_dropout = nn.Dropout(p=dropout_p)

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input to latent space."""
        h1 = F.relu(self.fc1(x))
        if self.bn1:
            h1 = self.bn1(h1)
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for sampling from latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Decode from latent space to reconstruction."""
        h3 = F.relu(self.fc3(z))
        if self.bn2:
            h3 = self.bn2(h3)
        logits = self.fc4(h3)
        return torch.sigmoid(logits / temperature)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> tuple:
        """Forward pass through Beta-VAE."""
        x_noisy = self.input_dropout(x)
        mu, logvar = self.encode(x_noisy)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, temperature=temperature)
        return recon, mu, logvar

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """Generate samples from the prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input without sampling."""
        _, mu, _ = self(x)
        return self.decode(mu)
