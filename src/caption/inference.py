from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction

from src.caption.evaluation import generate_captions_batch
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def run_caption_inference(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    predictions_path: Path,
    compute_perplexity: bool = False,
    compute_llm_judge: bool = False,
    llm_judge_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run inference on examples using batch processing with optional quality
    metrics.

    Args:
        cfg: Configuration
        model: Language model
        tokenizer: Tokenizer
        examples: List of examples to process
        predictions_path: Path to save results
        compute_perplexity: Whether to compute perplexity for each generated caption
        compute_llm_judge: Whether to evaluate with LLM-as-a-judge
        llm_judge_config: Configuration for LLM judge (model_name, device, etc.)

    Returns:
        DataFrame with predictions
    """
    # Extract prompts and aspects from examples
    prompts = [example[cfg.data.text_column] for example in examples]
    aspects = [example.get(cfg.data.aspect_column, "") for example in examples]
    ids = [
        example.get("id", f"sample_{i}") for i, example in enumerate(examples)
    ]

    # Generate captions in batches
    if log:
        log.info(
            f"Running inference on {len(prompts)} examples with batch size {cfg.evaluation.batch_size}..."
        )

    predictions, quality_metrics = generate_captions_batch(
        model,
        tokenizer,
        prompts,
        cfg.generation,
        batch_size=cfg.evaluation.batch_size,
        compute_perplexity=compute_perplexity,
        compute_llm_judge=compute_llm_judge,
        llm_judge_config=llm_judge_config,
    )

    # Prepare output records
    records = [
        {
            "id": id_val,
            "aspect_list": aspect,
            "prediction": prediction,
        }
        for id_val, aspect, prediction in zip(ids, aspects, predictions)
    ]

    # Add perplexity scores if computed
    if quality_metrics and "perplexity" in quality_metrics:
        perplexity_scores = quality_metrics["perplexity"].get("all_scores", [])
        if len(perplexity_scores) == len(records):
            for i, record in enumerate(records):
                record["perplexity"] = perplexity_scores[i]

    # Add LLM judge scores if computed
    if quality_metrics and "llm_judge" in quality_metrics:
        llm_scores = quality_metrics["llm_judge"].get("llm_judge_scores", [])
        llm_reasonings = quality_metrics["llm_judge"].get(
            "llm_judge_reasonings", []
        )
        if len(llm_scores) == len(records):
            for i, record in enumerate(records):
                record["llm_judge_score"] = llm_scores[i]
                record["llm_judge_reasoning"] = (
                    llm_reasonings[i] if i < len(llm_reasonings) else ""
                )

    # Save results
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(records)
    results_df.to_csv(predictions_path, index=False)

    if log:
        log.info(
            f"Inference completed. Predictions saved to {predictions_path}"
        )

    # Save quality metrics if computed
    if quality_metrics:
        metrics_path = predictions_path.parent / "quality_metrics.json"
        # Remove large lists from metrics for cleaner JSON
        metrics_to_save = {}
        for key, value in quality_metrics.items():
            if isinstance(value, dict):
                metrics_to_save[key] = {
                    k: v
                    for k, v in value.items()
                    if k
                    not in [
                        "all_scores",
                        "llm_judge_scores",
                        "llm_judge_reasonings",
                    ]
                }
            else:
                metrics_to_save[key] = value

        with open(metrics_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        if log:
            log.info(f"Quality metrics saved to {metrics_path}")

    return results_df
