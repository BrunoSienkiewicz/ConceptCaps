import hydra
from pathlib import Path
import torch
import rootutils
import lightning as pl
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

@hydra.main(version_base=None, config_path="../../../config", config_name="caption_evaluation")
def main(cfg: CaptionGenerationConfig) -> None:
    """Main evaluation function using PyTorch Lightning."""
    assert cfg.model.checkpoint_dir
    
    # Set random seed for reproducibility
    log.info(f"Setting random seed to {cfg.random_state}...")
    pl.seed_everything(cfg.random_state, workers=True)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Print configuration
    print_config_tree(cfg)

    # Load and prepare datasets
    log.info("Loading datasets...")
    dataset = load_dataset(cfg.data.dataset_name)
    log.info("Preparing datasets...")
    dataset = prepare_datasets(cfg.data, cfg.prompt, dataset)
    log.info(f"Test examples count: {len(dataset['test'])}")

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
        optimizer_cfg=cfg.trainer.optimizer,
        lr_scheduler_cfg=cfg.trainer.lr_scheduler,
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
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        callbacks=callbacks,
        logger=loggers,
    )

    # Test model
    log.info("Running evaluation...")
    trainer.test(
        model=model,
        datamodule=datamodule,
    )

if __name__ == "__main__":
    main()