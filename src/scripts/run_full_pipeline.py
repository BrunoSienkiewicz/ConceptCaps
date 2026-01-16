"""
Full Pipeline: Fine-tuning → Inference → TTA Audio Generation

This script combines three stages into a single pipeline:
1. Fine-tune a caption generation model (LLaMA/Qwen/etc.)
2. Run inference to generate captions from the fine-tuned model
3. Generate audio samples from the captions using MusicGen

Can be run on SLURM with the provided shell script.
"""

from __future__ import annotations

import json
import os
import torch
import hydra
import rootutils
import lightning as pl
import pandas as pd
import scipy.io.wavfile
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import print_config_tree, RankedLogger, instantiate_loggers, instantiate_callbacks
from src.caption.data import prepare_datasets, prepare_inference_datasets
from src.caption.modeling import prepare_tokenizer
from src.caption.evaluation import MetricComputer, generate_captions_batch
from src.caption.lightning_module import CaptionFineTuningModule
from src.caption.lightning_datamodule import CaptionDataModule
from src.tta.data import TTADataset
from src.tta.modelling import prepare_model as prepare_tta_model, prepare_tokenizer as prepare_tta_tokenizer


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


class PipelineStage:
    """Enum-like class for pipeline stages."""
    FINE_TUNING = "fine_tuning"
    INFERENCE = "inference"
    TTA_GENERATION = "tta_generation"


def run_fine_tuning(
    cfg: DictConfig,
    checkpoint_dir: Path,
    callbacks: list,
    loggers: list,
) -> CaptionFineTuningModule:
    """Run the fine-tuning stage.
    
    Args:
        cfg: Pipeline configuration
        checkpoint_dir: Directory to save model checkpoints
        callbacks: Lightning callbacks
        loggers: Lightning loggers
        
    Returns:
        Trained model
    """
    log.info("=" * 60)
    log.info("STAGE 1: Fine-tuning Caption Model")
    log.info("=" * 60)

    # Load and prepare datasets
    log.info(f"Loading dataset {cfg.fine_tuning.data.dataset_name}...")
    dataset = load_dataset(cfg.fine_tuning.data.dataset_name)
    log.info("Preparing datasets...")
    dataset = prepare_datasets(cfg.fine_tuning.data, cfg.fine_tuning.prompt, dataset)
    log.info(
        f"Dataset loaded with {len(dataset['train'])} training "
        f"and {len(dataset['validation'])} validation samples."
    )

    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.fine_tuning.model)

    # Create metric computer
    metric_computer = MetricComputer(cfg.fine_tuning.evaluation.metrics, tokenizer)

    # Create Lightning DataModule
    log.info("Creating DataModule...")
    datamodule = CaptionDataModule(
        dataset=dataset,
        tokenizer=tokenizer,
        data_cfg=cfg.fine_tuning.data,
        prompt_cfg=cfg.fine_tuning.prompt,
        batch_size=cfg.fine_tuning.data.batch_size,
        num_workers=cfg.fine_tuning.data.dataloader_num_workers,
        max_length=cfg.fine_tuning.generation.max_length,
    )

    # Init Lightning Module
    log.info("Creating Lightning Module...")
    model = CaptionFineTuningModule(
        model_cfg=cfg.fine_tuning.model,
        generation_cfg=cfg.fine_tuning.generation,
        lora_cfg=cfg.fine_tuning.lora,
        optimizer_cfg=cfg.fine_tuning.trainer.optimizer,
        lr_scheduler_cfg=cfg.fine_tuning.trainer.lr_scheduler,
        prompt_cfg=cfg.fine_tuning.prompt,
        tokenizer=tokenizer,
        metric_computer=metric_computer,
    )

    # Create Trainer
    log.info("Creating Lightning Trainer...")
    trainer = pl.Trainer(
        default_root_dir=str(checkpoint_dir),
        max_epochs=cfg.fine_tuning.trainer.max_epochs,
        accelerator=cfg.fine_tuning.trainer.accelerator,
        devices=cfg.fine_tuning.trainer.devices,
        strategy=cfg.fine_tuning.trainer.strategy,
        precision=cfg.fine_tuning.trainer.precision,
        gradient_clip_val=cfg.fine_tuning.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.fine_tuning.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.fine_tuning.trainer.log_every_n_steps,
        val_check_interval=cfg.fine_tuning.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.fine_tuning.trainer.check_val_every_n_epoch,
        enable_progress_bar=cfg.fine_tuning.trainer.enable_progress_bar,
        enable_model_summary=cfg.fine_tuning.trainer.enable_model_summary,
        deterministic=cfg.fine_tuning.trainer.deterministic,
        callbacks=callbacks,
        logger=loggers,
    )

    # Train the model
    log.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # Test model
    if cfg.fine_tuning.get("run_test", True):
        log.info("Running evaluation on test set...")
        trainer.test(model=model, datamodule=datamodule)

    # Save the adapter weights separately (for LoRA)
    if hasattr(model.model, "save_pretrained"):
        adapter_path = checkpoint_dir / "lora_adapter"
        model.model.save_pretrained(adapter_path)
        log.info(f"Saved LoRA adapter to {adapter_path}")

    log.info(f"Fine-tuning completed. Checkpoints saved to {checkpoint_dir}")
    
    return model


def run_inference(
    cfg: DictConfig,
    model: Optional[CaptionFineTuningModule],
    checkpoint_dir: Path,
    output_dir: Path,
    loggers: list,
) -> pd.DataFrame:
    """Run the inference stage.
    
    Args:
        cfg: Pipeline configuration
        model: Pre-loaded model (if available from fine-tuning stage)
        checkpoint_dir: Directory containing model checkpoints
        output_dir: Directory to save inference outputs
        loggers: Lightning loggers
        
    Returns:
        DataFrame containing generated captions
    """
    log.info("=" * 60)
    log.info("STAGE 2: Caption Inference")
    log.info("=" * 60)

    device = torch.device(cfg.device)
    
    # Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.inference.model)

    # Load or reuse model
    if model is None:
        log.info(f"Loading model from checkpoint: {checkpoint_dir}...")
        model = CaptionFineTuningModule(
            model_cfg=cfg.inference.model,
            generation_cfg=cfg.inference.generation,
            lora_cfg=cfg.inference.lora,
            optimizer_cfg=cfg.inference.trainer.optimizer,
            prompt_cfg=cfg.inference.prompt,
            lr_scheduler_cfg=cfg.inference.trainer.lr_scheduler,
            tokenizer=tokenizer,
        )
    else:
        log.info("Using model from fine-tuning stage...")

    model.eval()
    model.to(device)
    log.info("Model loaded successfully.")

    # Load and prepare inference dataset
    log.info(f"Loading inference dataset {cfg.inference.data.dataset_name}...")
    dataset = load_dataset(cfg.inference.data.dataset_name)
    log.info("Preparing datasets for inference...")
    dataset = prepare_inference_datasets(cfg.inference.data, cfg.inference.prompt, dataset)
    log.info(f"Dataset loaded with splits: {list(dataset.keys())}")

    all_records = []

    # Run inference for each split
    for split in dataset.keys():
        log.info(f"Running inference on '{split}' split...")
        
        examples = dataset[split]
        
        # Apply sample limits based on split
        max_samples_key = f"max_{split}_samples"
        max_samples = cfg.inference.data.get(max_samples_key, None)
        if max_samples:
            examples = examples.select(range(min(max_samples, len(examples))))
            log.info(f"Limiting to first {max_samples} samples for {split}.")
        
        # Extract prompts and aspects
        ids = [example[cfg.inference.data.id_column] for example in examples]
        prompts = [example[cfg.inference.data.text_column] for example in examples]
        aspects = [example.get(cfg.inference.data.aspect_column, "") for example in examples]
        
        log.info(f"Generating captions for {len(prompts)} examples...")
        
        # Generate captions in batches
        compute_perplexity = cfg.inference.evaluation.get("compute_perplexity", False)
        compute_llm_judge = cfg.inference.evaluation.get("compute_llm_judge", False)
        
        predictions, quality_metrics = generate_captions_batch(
            model,
            tokenizer,
            prompts,
            generate_cfg=cfg.inference.generation,
            batch_size=cfg.inference.data.batch_size,
            compute_perplexity=compute_perplexity,
            compute_llm_judge=compute_llm_judge,
            llm_judge_config=cfg.inference.evaluation.get("llm_judge_config", None),
        )
        
        # Prepare output records
        records = [
            {
                "id": id_,
                "aspect_list": aspect,
                "caption": prediction,  # Use 'caption' for TTA compatibility
                "prediction": prediction,
                "split": split,
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
        
        all_records.extend(records)
        
        # Save split-specific results
        predictions_path = output_dir / f"{split}_predictions.csv"
        results_df = pd.DataFrame(records)
        results_df.to_csv(predictions_path, index=False)
        log.info(f"Saved {len(results_df)} predictions to: {predictions_path}")
        
        # Log quality metrics
        if quality_metrics:
            metrics_path = output_dir / f"{split}_quality_metrics.json"
            metrics_to_save = {}
            for key, value in quality_metrics.items():
                if isinstance(value, dict):
                    metrics_to_save[f"{split}/{key}"] = {
                        k: v for k, v in value.items()
                        if k not in ['all_scores', 'llm_judge_scores', 'llm_judge_reasonings']
                    }
                else:
                    metrics_to_save[f"{split}/{key}"] = value
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_to_save, f, indent=2)
            log.info(f"Quality metrics saved to {metrics_path}")

            for logger in loggers:
                if hasattr(logger, 'log_metrics'):
                    logger.log_metrics(quality_metrics)

    # Combine all results
    combined_df = pd.DataFrame(all_records)
    combined_path = output_dir / "all_predictions.csv"
    combined_df.to_csv(combined_path, index=False)
    log.info(f"Combined predictions saved to: {combined_path}")
    
    log.info(f"Inference completed! Total {len(combined_df)} captions generated.")
    
    return combined_df


def run_tta_generation(
    cfg: DictConfig,
    captions_df: pd.DataFrame,
    output_dir: Path,
    loggers: list,
) -> None:
    """Run the TTA (Text-to-Audio) generation stage.
    
    Args:
        cfg: Pipeline configuration
        captions_df: DataFrame containing generated captions
        output_dir: Directory to save audio outputs
        loggers: Lightning loggers
    """
    log.info("=" * 60)
    log.info("STAGE 3: Text-to-Audio Generation")
    log.info("=" * 60)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Filter captions based on split configuration
    target_split = cfg.tta.get("target_split", "test")
    if "split" in captions_df.columns:
        captions_df = captions_df[captions_df["split"] == target_split].reset_index(drop=True)
        log.info(f"Using {len(captions_df)} captions from '{target_split}' split.")
    
    # Optional: limit number of samples
    max_samples = cfg.tta.get("max_samples", None)
    if max_samples and len(captions_df) > max_samples:
        captions_df = captions_df.head(max_samples).reset_index(drop=True)
        log.info(f"Limited to {max_samples} samples.")

    # Prepare TTA model and tokenizer
    log.info(f"Loading TTA model: {cfg.tta.model.name}...")
    processor = prepare_tta_tokenizer(cfg.tta.model)
    model = prepare_tta_model(cfg.tta.model)
    
    # Tokenize captions
    caption_column = cfg.tta.data.get("caption_column", "caption")
    captions = captions_df[caption_column].tolist()
    
    log.info(f"Tokenizing {len(captions)} captions...")
    inputs = processor(
        text=captions,
        max_length=cfg.tta.data.get("max_sequence_length", 256),
        padding=True,
        return_tensors="pt",
        truncation=True,
    )
    
    # Create dataset and dataloader
    tta_dataset = TTADataset(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        device=torch.device("cpu"),  # Move to device in batch
    )
    
    dataloader = DataLoader(
        dataset=tta_dataset,
        batch_size=cfg.tta.data.get("batch_size", 4),
        shuffle=False,
    )
    
    # Setup output directories
    audio_dir = output_dir / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Generation parameters
    id_column = cfg.tta.data.get("id_column", "id")
    filename_template = cfg.tta.data.get("filename_template", "{}.wav")
    max_new_tokens = cfg.tta.model.tokenizer.get("max_new_tokens", 256)
    batch_size = cfg.tta.data.get("batch_size", 4)
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": cfg.tta.generation.get("temperature", 1.0),
        "top_k": cfg.tta.generation.get("top_k", 50),
        "top_p": cfg.tta.generation.get("top_p", 0.95),
        "do_sample": cfg.tta.generation.get("do_sample", True),
        "use_cache": True,
    }
    
    guidance_scale = cfg.tta.generation.get("guidance_scale", None)
    if guidance_scale is not None:
        generation_kwargs["guidance_scale"] = guidance_scale
    
    log.info("Generating audio samples...")
    
    # Check if using accelerator for multi-GPU
    use_accelerator = cfg.tta.generation.get("use_accelerator", False)
    
    if use_accelerator:
        log.info("Using Accelerate for distributed generation...")
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        model, dataloader = accelerator.prepare(model, dataloader)
        
        # Get unwrapped model
        unwrapped_model = model.module if hasattr(model, 'module') else model
        sample_rate = cfg.tta.generation.get("sample_rate", unwrapped_model.config.audio_encoder.sampling_rate)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating audio [GPU {accelerator.process_index}]", 
                                                disable=not accelerator.is_main_process)):
            input_ids, attention_mask = batch
            
            with torch.inference_mode():
                audio_values = unwrapped_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            
            audio_values = accelerator.gather(audio_values)
            
            if accelerator.is_main_process:
                for item_idx, audio in enumerate(audio_values):
                    global_idx = batch_idx * batch_size * accelerator.num_processes + item_idx
                    if global_idx < len(captions_df):
                        sample_id = captions_df.iloc[global_idx][id_column]
                        scipy.io.wavfile.write(
                            audio_dir / filename_template.format(sample_id),
                            sample_rate,
                            audio[0].float().cpu().numpy(),
                        )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
    else:
        # Single GPU generation
        model = model.to(device)
        sample_rate = model.config.audio_encoder.sampling_rate
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating audio")):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            with torch.inference_mode():
                audio_values = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            
            for item_idx, audio in enumerate(audio_values):
                global_idx = batch_idx * batch_size + item_idx
                if global_idx < len(captions_df):
                    sample_id = captions_df.iloc[global_idx][id_column]
                    scipy.io.wavfile.write(
                        audio_dir / filename_template.format(sample_id),
                        sample_rate,
                        audio[0].float().cpu().numpy(),
                    )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save metadata
    captions_df["filename"] = captions_df[id_column].apply(lambda x: filename_template.format(x))
    metadata_path = output_dir / "metadata.csv"
    captions_df.to_csv(metadata_path, index=False)
    log.info(f"Metadata saved to {metadata_path}")
    
    # Run evaluation if configured
    if not cfg.tta.evaluation.get("skip_evaluation", True):
        log.info("Running TTA evaluation...")
        from src.tta.evaluate import TTAEvaluator
        
        evaluator = TTAEvaluator(
            clap_model=cfg.tta.evaluation.get("clap_model", "laion/clap-htsat-unfused"),
            fad_model=cfg.tta.evaluation.get("fad_model", "google/vggish"),
            device=str(device),
        )
        
        results = evaluator.evaluate(
            generated_audio_dir=audio_dir,
            metadata_path=metadata_path,
            output_dir=output_dir / "evaluation_results",
            text_column=caption_column,
            filename_column="filename",
            batch_size=cfg.tta.data.get("batch_size", 8),
        )
        
        for logger in loggers:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(results, step=0)
        
        log.info(f"Evaluation results saved to {output_dir / 'evaluation_results'}")
    
    log.info(f"TTA generation completed! Audio saved to: {audio_dir}")


@hydra.main(version_base=None, config_path="../../config", config_name="full_pipeline")
def main(cfg: DictConfig) -> None:
    """Main pipeline function combining fine-tuning, inference, and TTA generation."""
    
    # Set random seed for reproducibility
    log.info(f"Setting random seed to {cfg.random_state}...")
    pl.seed_everything(cfg.random_state, workers=True)

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    torch.set_float32_matmul_precision("medium")

    # Print configuration
    print_config_tree(cfg)

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

    # Setup directories
    base_output_dir = Path(cfg.paths.output_dir) / cfg.run_id
    model_dir = Path(cfg.paths.model_dir)
    data_dir = Path(cfg.paths.data_dir)
    
    # Stage-specific directories
    checkpoint_dir = model_dir / cfg.fine_tuning.model.name / cfg.run_id
    inference_output_dir = data_dir / "generated_captions" / cfg.run_id
    tta_output_dir = data_dir / "tta_outputs" / cfg.run_id
    
    # Create directories
    for dir_path in [base_output_dir, checkpoint_dir, inference_output_dir, tta_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which stages to run
    stages_to_run = cfg.get("stages", [
        PipelineStage.FINE_TUNING,
        PipelineStage.INFERENCE,
        PipelineStage.TTA_GENERATION,
    ])
    
    log.info(f"Pipeline stages to run: {stages_to_run}")
    
    # Initialize variables for cross-stage data passing
    fine_tuned_model = None
    captions_df = None
    
    # STAGE 1: Fine-tuning
    if PipelineStage.FINE_TUNING in stages_to_run:
        fine_tuned_model = run_fine_tuning(
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            callbacks=callbacks,
            loggers=loggers,
        )
    
    # STAGE 2: Inference
    if PipelineStage.INFERENCE in stages_to_run:
        # Update model checkpoint path if we just finished fine-tuning
        if fine_tuned_model is not None:
            # Use the model from fine-tuning directly
            pass
        elif cfg.inference.model.get("checkpoint_dir"):
            # Use specified checkpoint
            log.info(f"Using checkpoint from: {cfg.inference.model.checkpoint_dir}")
        else:
            # Default to the checkpoint directory from this run
            cfg.inference.model.checkpoint_dir = str(checkpoint_dir / "lora_adapter")
        
        captions_df = run_inference(
            cfg=cfg,
            model=fine_tuned_model,
            checkpoint_dir=checkpoint_dir,
            output_dir=inference_output_dir,
            loggers=loggers,
        )
    elif PipelineStage.TTA_GENERATION in stages_to_run:
        # Load captions from a previous run if skipping inference
        captions_path = cfg.get("captions_path", None)
        if captions_path:
            log.info(f"Loading captions from: {captions_path}")
            captions_df = pd.read_csv(captions_path)
        else:
            # Try to load from default location
            default_path = inference_output_dir / "all_predictions.csv"
            if default_path.exists():
                log.info(f"Loading captions from: {default_path}")
                captions_df = pd.read_csv(default_path)
            else:
                raise ValueError(
                    "No captions available for TTA generation. "
                    "Either run inference stage or provide 'captions_path' in config."
                )
    
    # STAGE 3: TTA Generation
    if PipelineStage.TTA_GENERATION in stages_to_run:
        if captions_df is None:
            raise ValueError("No captions available for TTA generation.")
        
        run_tta_generation(
            cfg=cfg,
            captions_df=captions_df,
            output_dir=tta_output_dir,
            loggers=loggers,
        )
    
    # Save final dataset summary
    summary = {
        "run_id": cfg.run_id,
        "stages_completed": stages_to_run,
        "checkpoint_dir": str(checkpoint_dir),
        "inference_output_dir": str(inference_output_dir),
        "tta_output_dir": str(tta_output_dir),
        "total_captions": len(captions_df) if captions_df is not None else 0,
    }
    
    summary_path = base_output_dir / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log.info("=" * 60)
    log.info("PIPELINE COMPLETED SUCCESSFULLY!")
    log.info("=" * 60)
    log.info(f"Summary saved to: {summary_path}")
    log.info(f"Model checkpoints: {checkpoint_dir}")
    log.info(f"Generated captions: {inference_output_dir}")
    log.info(f"Generated audio: {tta_output_dir}")


if __name__ == "__main__":
    main()
