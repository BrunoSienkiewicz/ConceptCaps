from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import evaluate
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction
from tqdm import tqdm

from src.utils import RankedLogger
from src.caption.evaluation import generate_captions_batch


def run_caption_inference(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[dict],
    output_dir: Path,
    logger: Optional[RankedLogger] = None
) -> pd.DataFrame:
    """
    Run inference on examples using batch processing.
    
    Args:
        cfg: Configuration
        model: Language model
        tokenizer: Tokenizer
        examples: List of examples to process
        output_dir: Directory to save results
        logger: Logger instance
        batch_size: Batch size for generation
        
    Returns:
        DataFrame with predictions
    """
    # Extract prompts and aspects from examples
    prompts = [example[cfg.data.text_column] for example in examples]
    aspects = [example.get(cfg.data.aspect_column, "") for example in examples]

    # Generate captions in batches
    if logger:
        logger.info(
            f"Running inference on {len(prompts)} examples with batch size {cfg.evaluation.batch_size}..."
        )
    predictions = generate_captions_batch(
        model,
        tokenizer,
        prompts,
        cfg.evaluation.max_new_tokens,
        batch_size=cfg.evaluation.batch_size,
    )

    # Prepare output records
    records = [
        {
            "aspect_list": aspect,
            "prediction": prediction,
        }
        for aspect, prediction in zip(aspects, predictions)
    ]

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / cfg.evaluation.predictions_file
    results_df = pd.DataFrame(records)
    results_df.to_csv(predictions_path, index=False)
    
    if logger:
        logger.info(f"Inference completed. Predictions saved to {predictions_path}")

    return results_df
    