from __future__ import annotations

import torch
import hydra
import rootutils
import lightning as pl
from pathlib import Path
from datasets import load_dataset

from src.caption import CaptionGenerationConfig
from src.utils import print_config_tree, RankedLogger, instantiate_loggers, instantiate_callbacks
from src.caption.data import prepare_datasets
from src.caption.modeling import prepare_tokenizer
from src.caption.evaluation import MetricComputer
from src.caption.lightning_module import CaptionFineTuningModule
from src.caption.lightning_datamodule import CaptionDataModule


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="caption_fine_tuning")
def main(cfg: CaptionGenerationConfig) -> None:
    """Main training function using PyTorch Lightning."""
    
    # Set random seed for reproducibility
    log.info(f"Setting random seed to {cfg.random_state}...")
    pl.seed_everything(cfg.random_state, workers=True)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Print configuration
    print_config_tree(cfg)

    # Setup directories
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / cfg.model.name / cfg.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare datasets
    log.info("Loading datasets...")
    dataset = load_dataset(cfg.data.dataset_name)
    log.info("Preparing datasets...")
    dataset = prepare_datasets(cfg.data, cfg.prompt, dataset)
    log.info(
        f"Dataset loaded with {len(dataset['train'])} training "
        f"and {len(dataset['validation'])} validation samples."
    )

    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    # Create metric computer
    metric_computer = MetricComputer(cfg.evaluation.metrics, tokenizer)

    # Create Lightning DataModule
    log.info("Creating DataModule...")
    datamodule = CaptionDataModule(
        dataset=dataset,
        tokenizer=tokenizer,
        data_cfg=cfg.data,
        batch_size=cfg.trainer.per_device_train_batch_size,
        num_workers=cfg.trainer.get("dataloader_num_workers", 4),
        max_length=cfg.data.get("max_length", 512),
    )

    # Create Lightning Module
    log.info("Creating Lightning Module...")
    model = CaptionFineTuningModule(
        model_cfg=cfg.model,
        lora_cfg=cfg.lora,
        optimizer_cfg=cfg.trainer,
        tokenizer=tokenizer,
        metric_computer=metric_computer,
    )

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

    # Create Trainer
    log.info("Creating Lightning Trainer...")
    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_dir),
        max_epochs=cfg.trainer.num_train_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto" if cfg.trainer.get("ddp", False) else "auto",
        precision=cfg.trainer.get("precision", "16-mixed"),
        gradient_clip_val=cfg.trainer.get("max_grad_norm", 1.0),
        accumulate_grad_batches=cfg.trainer.get("gradient_accumulation_steps", 1),
        log_every_n_steps=cfg.trainer.get("logging_steps", 10),
        val_check_interval=cfg.trainer.get("eval_steps", None),
        check_val_every_n_epoch=1 if cfg.trainer.get("eval_steps") is None else None,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=cfg.get("deterministic", False),
    )

    # Train the model
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Save final model
    log.info("Saving final model...")
    final_model_path = checkpoint_dir / "final_model"
    trainer.save_checkpoint(final_model_path / "checkpoint.ckpt")
    
    # Save the adapter weights separately (for LoRA)
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(final_model_path)
        log.info(f"Saved LoRA adapter to {final_model_path}")
    
    log.info(f"Training completed. Model and checkpoints are saved in {checkpoint_dir}")


if __name__ == "__main__":
    main()