from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import evaluate
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction

from src.utils import RankedLogger


def generate_caption(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class MetricComputer:
    def __init__(self, metric_cfgs: Iterable[DictConfig], tokenizer: Optional[AutoTokenizer] = None):
        self.metric_cfgs = metric_cfgs
        self.metrics = []
        for metric_cfg in metric_cfgs:
            metric = evaluate.load(metric_cfg.name)
            self.metrics.append((metric, metric_cfg))
        self.tokenizer = tokenizer

    def _calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for metric, metric_cfg in self.metrics:
            kwargs = metric_cfg.get("kwargs", {})
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

    def compute_metrics(self, eval_pred: EvalPrediction):
        predictions = eval_pred.predictions
        references = eval_pred.label_ids

        results: Dict[str, Any] = {}
        if self.tokenizer is not None:
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(references, skip_special_tokens=True)
            results = self._calculate_metrics(decoded_preds, decoded_labels)
            
        return results

    def compute_test_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        results = self._calculate_metrics(predictions, references)
        return results


def run_test_evaluation(
    cfg: DictConfig,
    computer: MetricComputer,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_examples: List[dict],
    output_dir: Path,
    logger: Optional[RankedLogger] = None,
) -> Dict[str, Any]:
    predictions: List[str] = []
    references: List[str] = []
    records: List[Dict[str, Any]] = []

    for example in eval_examples:
        generated = generate_caption(
            model,
            tokenizer,
            example,
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

    metrics = computer.compute_test_metrics(predictions, references)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if cfg.evaluation.output_predictions:
        predictions_path = output_dir / cfg.evaluation.predictions_file
        pd.DataFrame(records).to_csv(predictions_path, index=False)

    if logger:
        logger.info(f"Evaluation metrics: {metrics}")

    return metrics
