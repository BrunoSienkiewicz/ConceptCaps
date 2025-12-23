"""VAE training script using PyTorch Lightning with Hydra configuration."""
from __future__ import annotations

import torch
import numpy as np
import hydra
import rootutils
import lightning as pl
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from src.vae import VAELightningModule, BetaVAELightningModule, VAEDataModule
from src.vae.config import VAEConfig
from src.vae.metrics_saver import MetricsSaver, MetricsSaveCallback
from src.utils import print_config_tree, RankedLogger, instantiate_loggers, instantiate_callbacks


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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
    from src.vae.metrics import VAEMetrics
    
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


@hydra.main(version_base=None, config_path="../../../config", config_name="vae_training")
def main(cfg: DictConfig) -> None:
    """Main training function for VAE using PyTorch Lightning."""
    
    # Convert config to typed config class
    cfg = VAEConfig(**cfg) if isinstance(cfg, dict) else cfg
    
    # Setup callbacks
    callbacks = []
    if cfg.get("callbacks"):
        pl_callbacks = instantiate_callbacks(cfg.callbacks)
        if pl_callbacks:
            callbacks.extend(pl_callbacks if isinstance(pl_callbacks, list) else [pl_callbacks])
    
    # Setup loggers
    loggers = []
    if cfg.get("logger"):
        pl_loggers = instantiate_loggers(cfg.logger)
        if pl_loggers:
            loggers.extend(pl_loggers if isinstance(pl_loggers, list) else [pl_loggers])

    if loggers:
        for logger in loggers:
            if hasattr(logger, 'experiment'):
                logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    
    # Set random seed for reproducibility
    log.info(f"Setting random seed to {cfg.random_state}...")
    pl.seed_everything(cfg.random_state, workers=True)
    
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")
    
    torch.set_float32_matmul_precision("medium")
    
    # Print configuration
    print_config_tree(cfg)
    
    # Setup directories
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / cfg.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Lightning DataModule
    log.info("Creating DataModule...")
    datamodule = VAEDataModule(
        data_cfg=cfg.data,
        taxonomy_path=cfg.data.taxonomy_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.dataloader_num_workers,
    )
    
    # Setup to compute input_dim
    datamodule.setup()
    cfg.model.input_dim = datamodule.total_input_dim
    log.info(f"Total input dimension: {cfg.model.input_dim}")
    
    # Init Lightning Module
    log.info("Creating Lightning Module...")
    
    # Detect if Beta-VAE based on model name or config
    use_beta_vae = "beta" in cfg.model_name.lower() or (
        hasattr(cfg.model, 'beta') and cfg.model.beta is not None and cfg.model.beta != 1.0
    )
    
    if use_beta_vae:
        log.info(f"Using Beta-VAE with beta={getattr(cfg.model, 'beta', 4.0)}")
        model = BetaVAELightningModule(
            model_cfg=cfg.model,
            loss_cfg=cfg.loss,
            learning_rate=cfg.trainer.optimizer.lr,
            betas=cfg.trainer.optimizer.betas if hasattr(cfg.trainer.optimizer, 'betas') else (0.9, 0.999),
            weight_decay=cfg.trainer.optimizer.weight_decay if hasattr(cfg.trainer.optimizer, 'weight_decay') else 0.0,
            beta=cfg.model.beta if hasattr(cfg.model, 'beta') else 4.0,
        )
    else:
        log.info("Using standard VAE")
        model = VAELightningModule(
            model_cfg=cfg.model,
            loss_cfg=cfg.loss,
            learning_rate=cfg.trainer.optimizer.lr,
            betas=cfg.trainer.optimizer.betas if hasattr(cfg.trainer.optimizer, 'betas') else (0.9, 0.999),
            weight_decay=cfg.trainer.optimizer.weight_decay if hasattr(cfg.trainer.optimizer, 'weight_decay') else 0.0,
        )
    
    # Create metrics saver and add callback
    metrics_saver = MetricsSaver(checkpoint_dir)
    metrics_callback = MetricsSaveCallback(metrics_saver)
    callbacks.append(metrics_callback)
    
    # Create Trainer
    log.info("Creating Lightning Trainer...")
    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_dir),
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        deterministic=cfg.trainer.deterministic,
        callbacks=callbacks,
        logger=loggers,
    )
    
    # Train the model
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Save final model
    log.info("Saving final model...")
    final_model_path = checkpoint_dir / "final.ckpt"
    trainer.save_checkpoint(final_model_path)
    
    best_model_path = None
    for callback in callbacks:
        if isinstance(callback, pl.pytorch.callbacks.ModelCheckpoint):
            best_model_path = callback.best_model_path
            break
    
    if best_model_path:
        log.info(f"Best model checkpoint found at {best_model_path}")
    
    # Test model
    log.info("Running evaluation...")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )
    
    # Run inference on sampled latent vectors
    if cfg.inference.enabled:
        log.info("Running inference on sampled latent vectors...")
        inference_metrics = run_latent_inference(
            model=model,
            cfg=cfg,
            device=device,
            metrics_saver=metrics_saver,
        )
        for logger in loggers:
            if hasattr(logger, 'experiment'):
                for metric_name, metric_value in inference_metrics.items():
                    logger.experiment.log({metric_name: metric_value})
    
    # Save the model weights as PyTorch model (for inference)
    if cfg.save_model:
        log.info("Saving model weights...")
        model_save_path = model_dir / f"{cfg.model_name}.pth"
        torch.save(model.model.state_dict(), model_save_path)
        log.info(f"Model weights saved to {model_save_path}")
    
    # Save final metrics to models folder
    log.info("Saving final metrics...")
    metrics_dir = model_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics JSON
    metrics_file = metrics_saver.save(filename="metrics.json")
    log.info(f"Metrics saved to {metrics_file}")
    
    # Save summary with configuration
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    summary_file = metrics_saver.save_summary(
        config=config_dict,
        filename="summary.json"
    )
    log.info(f"Summary saved to {summary_file}")
    log.info(f"Training completed. Checkpoints and outputs are saved in {checkpoint_dir}")


if __name__ == "__main__":
    main()
