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
from caption.model import prepare_tokenizer
from src.caption.evaluation import MetricComputer
from src.caption.lightning_module import CaptionFineTuningModule
from src.caption.lightning_datamodule import CaptionDataModule


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="caption_fine_tuning")
def main(cfg: CaptionGenerationConfig) -> None:
    """Main training function using PyTorch Lightning."""

    callbacks = []
    if cfg.get("callbacks"):
        pl_callbacks = instantiate_callbacks(cfg.callbacks)
        if pl_callbacks:
            callbacks.extend(pl_callbacks if isinstance(pl_callbacks, list) else [pl_callbacks])

    loggers = []
    if cfg.get("logger"):
        pl_loggers = instantiate_loggers(cfg.logger)
        if pl_loggers:
            loggers.extend(pl_loggers if isinstance(pl_loggers, list) else [pl_loggers])

    log.info(f"Setting random seed to {cfg.random_state}...")
    pl.seed_everything(cfg.random_state, workers=True)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    torch.set_float32_matmul_precision("medium")

    print_config_tree(cfg)

    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / cfg.model.name / cfg.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        prompt_cfg=cfg.prompt,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.dataloader_num_workers,
        max_length=cfg.generation.max_length 
    )

    log.info("Creating Lightning Module...")
    model = CaptionFineTuningModule(
        model_cfg=cfg.model,
        generation_cfg=cfg.generation,
        lora_cfg=cfg.lora,
        optimizer_cfg=cfg.trainer.optimizer,
        lr_scheduler_cfg=cfg.trainer.lr_scheduler,
        prompt_cfg=cfg.prompt,
        tokenizer=tokenizer,
        metric_computer=metric_computer,
    )

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

    # Test model
    log.info("Running evaluation...")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )

    if cfg.evaluation.output_predictions:
        log.info("Saving evaluation predictions...")
        model.metric_computer.save_predictions(
            output_dir=checkpoint_dir,
        )

    # Save the adapter weights separately (for LoRA)
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(checkpoint_dir / "lora_adapter")
        log.info(f"Saved LoRA adapter to {checkpoint_dir / 'lora_adapter'}")
    
    log.info(f"Training completed. Model and checkpoints are saved in {checkpoint_dir}")


if __name__ == "__main__":
    main()