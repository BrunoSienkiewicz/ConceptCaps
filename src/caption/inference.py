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

from src.utils import RankedLogger
from src.caption.evaluation import generate_caption


def run_caption_inference(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: List[dict],
    output_dir: Path,
    logger: Optional[RankedLogger] = None,
):
    predictions: List[str] = []
    records: List[Dict[str, Any]] = []

    for example in examples:
        generated = generate_caption(
            model,
            tokenizer,
            prompt=example[cfg.data.text_column],
            max_new_tokens=cfg.evaluation.max_new_tokens,
        )
        predictions.append(generated)
        if logger:
            logger.info(f"Generated caption: {generated}")
        records.append(
            {
                "aspect_list": example[cfg.data.aspect_column],
                "prediction": generated,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / cfg.evaluation.predictions_file
    pd.DataFrame(records).to_csv(predictions_path, index=False)
    if logger:
        logger.info(f"Inference completed. Predictions saved to {predictions_path}")
    