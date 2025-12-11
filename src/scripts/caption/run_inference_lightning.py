from __future__ import annotations

import torch
import hydra
import rootutils
import lightning as pl
import pandas as pd
from pathlib import Path
from datasets import load_dataset

from src.caption import CaptionGenerationConfig
from src.utils import print_config_tree, RankedLogger, instantiate_loggers
from src.caption.data import prepare_inference_datasets
from src.caption.modeling import prepare_tokenizer
from src.caption.lightning_module import CaptionFineTuningModule
from src.caption.evaluation import generate_captions_batch


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="caption_inference")
def main(cfg: CaptionGenerationConfig) -> None:
    """Main inference function using PyTorch Lightning."""

    # Setup loggers
    loggers = []
    if cfg.get("logger"):
        pl_loggers = instantiate_loggers(cfg.logger)
        if pl_loggers:
            loggers.extend(pl_loggers if isinstance(pl_loggers, list) else [pl_loggers])
    
    # Set random seed for reproducibility
    log.info(f"Setting random seed to {cfg.random_state}...")
    pl.seed_everything(cfg.random_state, workers=True)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Print configuration
    print_config_tree(cfg)

    # Setup directories
    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir / cfg.model.name / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare datasets
    log.info(f"Loading dataset {cfg.data.dataset_name}...")
    dataset = load_dataset(cfg.data.dataset_name)
    log.info("Preparing datasets for inference...")
    dataset = prepare_inference_datasets(cfg.data, cfg.prompt, dataset)
    log.info(f"Dataset loaded with splits: {list(dataset.keys())}")

    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    # Load model from checkpoint
    log.info(f"Loading model from checkpoint: {cfg.model.checkpoint_dir}...")
    
    # Init Lightning Module
    log.info("Loading pretrained model for inference...")
    if cfg.model.checkpoint_dir:
        model = CaptionFineTuningModule.load_pretrained_model(
            model_cfg=cfg.model,
        )
    else:
        model = CaptionFineTuningModule(
            model_cfg=cfg.model,
            generation_cfg=cfg.generation,
            lora_cfg=cfg.lora,
            optimizer_cfg=cfg.trainer.optimizer,
            lr_scheduler_cfg=cfg.trainer.lr_scheduler,
            tokenizer=tokenizer,
        )

    
    model.eval()
    model.to(device)
    log.info("Model loaded successfully.")

    # Run inference for each split
    for split in dataset.keys():
        log.info(f"Running inference on '{split}' split...")
        
        examples = dataset[split]
        if cfg.data.max_test_samples and split == "test":
            examples = examples.select(range(cfg.data.max_test_samples))
            log.info(f"Limiting to first {cfg.data.max_test_samples} samples for testing.")
        elif cfg.data.max_val_samples and split == "validation":
            examples = examples.select(range(cfg.data.max_val_samples))
            log.info(f"Limiting to first {cfg.data.max_val_samples} samples for validation.")
        elif cfg.data.max_train_samples and split == "train":
            examples = examples.select(range(cfg.data.max_train_samples))
            log.info(f"Limiting to first {cfg.data.max_train_samples} samples for training.")
        
        # Extract prompts and aspects
        ids = [example[cfg.data.id_column] for example in examples]
        prompts = [example[cfg.data.text_column] for example in examples]
        aspects = [example.get(cfg.data.aspect_column, "") for example in examples]
        
        log.info(f"Generating captions for {len(prompts)} examples...")
        
        # Generate captions in batches
        predictions = generate_captions_batch(
            model,
            tokenizer,
            prompts,
            generate_cfg=cfg.generation,
            batch_size=cfg.data.batch_size,
        )
        
        # Prepare output records
        records = [
            {
                "id": id_,
                "aspect_list": aspect,
                "prediction": prediction,
            }
            for id_, aspect, prediction in zip(ids, aspects, predictions)
        ]
        
        # Save results
        predictions_path = output_dir / f"{split}_predictions.csv"
        results_df = pd.DataFrame(records)
        results_df.to_csv(predictions_path, index=False)
        
        log.info(f"Saved {len(results_df)} predictions to: {predictions_path}")
        
    log.info(f"Inference completed! All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
