from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import evaluate
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import RankedLogger


def generate_caption(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if model.dtype == torch.bfloat16:
        inputs = {key: value.bfloat16() for key, value in inputs.items()}
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_metrics(
    predictions: List[str],
    references: List[str],
    metric_cfgs: Iterable[DictConfig],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for metric_cfg in metric_cfgs:
        metric = evaluate.load(metric_cfg.name)
        kwargs = metric_cfg.get("kwargs", {}) or {}
        if metric_cfg.name == "bleu":
            value = metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references],
                **kwargs,
            )
        else:
            value = metric.compute(
                predictions=predictions,
                references=references,
                **kwargs,
            )
        results[metric_cfg.name] = value
    return results


def run_evaluation(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_examples: List[dict],
    output_dir: Path,
    logger: Optional[RankedLogger] = None,
) -> Dict[str, Any]:
    if not cfg.evaluation.enabled or not eval_examples:
        if logger:
            logger.info("Evaluation skipped.")
        return {}

    prompt_cfg = cfg.data.prompt
    prompt_template = cfg.evaluation.prompt_template

    predictions: List[str] = []
    references: List[str] = []
    records: List[Dict[str, Any]] = []

    for example in eval_examples:
        user_prompt = prompt_cfg["user_prompt_template"].format(
            tags=example[cfg.data.aspect_column]
        )
        prompt = prompt_template.format(
            system_prompt=prompt_cfg["system_prompt"],
            user_prompt=user_prompt,
        )
        generated = generate_caption(
            model,
            tokenizer,
            prompt,
            cfg.evaluation.max_new_tokens,
            cfg.evaluation.temperature,
        )
        predictions.append(generated)
        references.append(example[cfg.data.caption_column])
        records.append(
            {
                "aspect_list": example[cfg.data.aspect_column],
                "reference": example[cfg.data.caption_column],
                "prediction": generated,
            }
        )

    metrics = compute_metrics(predictions, references, cfg.evaluation.metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if cfg.evaluation.output_predictions:
        predictions_path = output_dir / cfg.evaluation.predictions_file
        pd.DataFrame(records).to_csv(predictions_path, index=False)

    if logger:
        logger.info(f"Evaluation metrics: {metrics}")

    return metrics
