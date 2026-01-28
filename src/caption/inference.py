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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.caption.evaluation import calculate_perplexity, evaluate_with_llm_judge
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def generate_captions_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int = 8,
    compute_perplexity: bool = False,
    compute_llm_judge: bool = False,
    llm_judge_config: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    """Generate captions for a batch of prompts efficiently with optional
    quality metrics.

    Args:
        model: Language model for generation
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of input prompts
        generate_cfg: Generation configuration
        batch_size: Number of prompts to process per batch
        compute_perplexity: Whether to compute perplexity for each generated caption
        compute_llm_judge: Whether to evaluate with LLM-as-a-judge
        llm_judge_config: Configuration for LLM judge (model_name, device, etc.)

    Returns:
        Tuple of (generated captions, quality metrics dict or None)
    """
    model.eval()
    captions = []
    perplexities = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(prompts), batch_size), desc="Generating captions"
        ):
            batch_prompts = prompts[i : i + batch_size]
            batch_captions = model.generate_captions_batch(batch_prompts)
            captions.extend(batch_captions)

            if compute_perplexity:
                for caption in batch_captions:
                    try:
                        ppl = calculate_perplexity(
                            model, tokenizer, caption, device=model.device
                        )
                        perplexities.append(ppl)
                    except Exception as e:
                        log.warning(f"Failed to calculate perplexity: {e}")
                        perplexities.append(float("inf"))

    # Compile quality metrics
    quality_metrics = {}

    if compute_perplexity:
        # Filter out infinite values for statistics
        valid_perplexities = [p for p in perplexities if p != float("inf")]
        if valid_perplexities:
            quality_metrics["perplexity"] = {
                "mean": float(np.mean(valid_perplexities)),
                "std": float(np.std(valid_perplexities)),
                "min": float(np.min(valid_perplexities)),
                "max": float(np.max(valid_perplexities)),
                "median": float(np.median(valid_perplexities)),
                "all_scores": perplexities,
            }
            log.info(
                f"Perplexity - Mean: {quality_metrics['perplexity']['mean']: .2f}, "
                f"Std: {quality_metrics['perplexity']['std']: .2f}"
            )

    if compute_llm_judge:
        if llm_judge_config is None:
            llm_judge_config = {
                "judge_model_name": "meta-llama/Llama-3.2-3B-Instruct",
                "batch_size": 4,
                "device": str(model.device),
            }

        llm_judge_results = evaluate_with_llm_judge(
            captions=captions,
            prompts=prompts,
            **llm_judge_config,
        )
        quality_metrics["llm_judge"] = llm_judge_results
        log.info(
            f"LLM Judge - Mean Score: {llm_judge_results['llm_judge_mean_score']: .2f}, "
            f"Std: {llm_judge_results['llm_judge_std_score']: .2f}"
        )

    return captions, quality_metrics
