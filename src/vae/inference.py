import torch
import numpy as np
from pathlib import Path
from src.utils import RankedLogger
from src.vae.metrics_saver import MetricsSaver


from vae.config import VAEConfig
from vae.lightning_module import VAELightningModule


log = RankedLogger(__name__, rank_zero_only=True)

def run_latent_inference(
    model: VAELightningModule,
    cfg: VAEConfig,
    device: torch.device,
    metrics_saver: MetricsSaver,
) -> None:
    """Run inference on sampled latent vectors and compute metrics.
    
    Args:
        model: Trained VAE Lightning module
        cfg: Configuration object
        device: Device to run inference on
        metrics_saver: MetricsSaver instance for logging metrics
    """
    from vae.evaluation import VAEMetrics
    
    model.to(device)
    model.eval()
    
    # Set seed for reproducibility
    if cfg.inference.seed is not None:
        torch.manual_seed(cfg.inference.seed)
        np.random.seed(cfg.inference.seed)
        log.info(f"Setting inference seed to {cfg.inference.seed}")
    
    log.info(f"Sampling {cfg.inference.num_samples} latent vectors from standard normal...")
    
    # Sample latent vectors from standard normal distribution
    z = torch.randn(cfg.inference.num_samples, model.model_cfg.latent_dim, device=device)
    
    with torch.no_grad():
        # Decode latent vectors to get reconstructions
        log.info(f"Decoding latent vectors with temperature={cfg.inference.temperature}...")
        recon = model.model.decode(z, temperature=cfg.inference.temperature)
        
        # Since we don't have ground truth for sampled latent vectors,
        # we compute generative quality metrics instead of reconstruction metrics
        
        # 1. Compute sparsity: how many tags are typically activated
        recon_binary = (recon > cfg.inference.threshold).float()
        avg_tags_per_sample = recon_binary.sum(dim=1).mean().item()
        std_tags_per_sample = recon_binary.sum(dim=1).std().item()
        min_tags_per_sample = recon_binary.sum(dim=1).min().item()
        max_tags_per_sample = recon_binary.sum(dim=1).max().item()
        
        # 2. Compute diversity: how diverse are the generated samples
        unique_combinations = torch.unique(recon_binary, dim=0).shape[0]
        diversity_ratio = unique_combinations / cfg.inference.num_samples
        
        # 3. Compute average probability distribution statistics
        mean_prob = recon.mean().item()
        std_prob = recon.std().item()
        entropy_per_sample = -(recon * torch.log(recon + 1e-8) + 
                               (1 - recon) * torch.log(1 - recon + 1e-8)).sum(dim=1).mean().item()
        
        # 4. Latent space statistics
        latent_stats = VAEMetrics.active_units(z, threshold=0.01)
        
        # Compile all metrics
        inference_metrics = {
            'inference/avg_tags_per_sample': avg_tags_per_sample,
            'inference/std_tags_per_sample': std_tags_per_sample,
            'inference/min_tags_per_sample': min_tags_per_sample,
            'inference/max_tags_per_sample': max_tags_per_sample,
            'inference/unique_combinations': unique_combinations,
            'inference/diversity_ratio': diversity_ratio,
            'inference/mean_probability': mean_prob,
            'inference/std_probability': std_prob,
            'inference/avg_entropy': entropy_per_sample,
            'inference/num_samples': cfg.inference.num_samples,
            'inference/temperature': cfg.inference.temperature,
            'inference/threshold': cfg.inference.threshold,
        }
        
        # Add latent space statistics
        for key, value in latent_stats.items():
            inference_metrics[f'inference/latent_{key}'] = value
        
        # Log metrics
        log.info("Latent Vector Inference Metrics:")
        metrics_saver.update("inference", inference_metrics)
        
    log.info("Latent vector inference completed.")

    return inference_metrics
