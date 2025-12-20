from __future__ import annotations

import json
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
    log.info("Loading model for inference...")
    model = CaptionFineTuningModule(
        model_cfg=cfg.model,
        generation_cfg=cfg.generation,
        lora_cfg=cfg.lora,
        optimizer_cfg=cfg.trainer.optimizer,
        prompt_cfg=cfg.prompt,
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
        
        # Generate captions in batches with optional quality metrics
        compute_perplexity = cfg.evaluation.get("compute_perplexity", False)
        compute_llm_judge = cfg.evaluation.get("compute_llm_judge", False)
        
        predictions, quality_metrics = generate_captions_batch(
            model,
            tokenizer,
            prompts,
            generate_cfg=cfg.generation,
            batch_size=cfg.data.batch_size,
            compute_perplexity=compute_perplexity,
            compute_llm_judge=compute_llm_judge,
            llm_judge_config=cfg.evaluation.get("llm_judge_config", None),
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
        
        # Add perplexity scores if computed
        if quality_metrics and 'perplexity' in quality_metrics:
            perplexity_scores = quality_metrics['perplexity'].get('all_scores', [])
            if len(perplexity_scores) == len(records):
                for i, record in enumerate(records):
                    record['perplexity'] = perplexity_scores[i]
        
        # Add LLM judge scores if computed
        if quality_metrics and 'llm_judge' in quality_metrics:
            llm_scores = quality_metrics['llm_judge'].get('llm_judge_scores', [])
            llm_reasonings = quality_metrics['llm_judge'].get('llm_judge_reasonings', [])
            if len(llm_scores) == len(records):
                for i, record in enumerate(records):
                    record['llm_judge_score'] = llm_scores[i]
                    record['llm_judge_reasoning'] = llm_reasonings[i] if i < len(llm_reasonings) else ""
        
        # Save results
        predictions_path = output_dir / f"{split}_predictions.csv"
        results_df = pd.DataFrame(records)
        results_df.to_csv(predictions_path, index=False)
        
        log.info(f"Saved {len(results_df)} predictions to: {predictions_path}")
        
        # Save quality metrics if computed
        if quality_metrics:
            metrics_path = output_dir / f"{split}_quality_metrics.json"
            # Remove large lists from metrics for cleaner JSON
            metrics_to_save = {}
            for key, value in quality_metrics.items():
                if isinstance(value, dict):
                    metrics_to_save[split + "/" + key] = {
                        k: v for k, v in value.items() 
                        if k not in ['all_scores', 'llm_judge_scores', 'llm_judge_reasonings']
                    }
                else:
                    metrics_to_save[split + "/" + key] = value
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            
            log.info(f"Quality metrics saved to {metrics_path}")

            for logger in loggers:
                if hasattr(logger, 'log_metrics'):
                    logger.log_metrics(quality_metrics, step=None)
        
    log.info(f"Inference completed! All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
