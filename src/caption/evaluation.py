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


logger = RankedLogger(__name__, rank_zero_only=True)

def generate_caption(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate a single caption from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
    # Extract only new tokens (skip input tokens)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_captions_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generate_cfg: DictConfig,
    batch_size: int = 8,
) -> List[str]:
    """
    Generate captions for a batch of prompts efficiently.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for encoding/decoding
        prompts: List of input prompts
        max_new_tokens: Maximum tokens to generate per caption
        batch_size: Number of prompts to process per batch
        
    Returns:
        List of generated captions
    """
    model.eval()
    captions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating captions"):
            batch_prompts = prompts[i : i + batch_size]
            
            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=generate_cfg.max_length,
            ).to(model.device)
            
            # Generate batch
            outputs = model.generate(
                **inputs,
                max_new_tokens=generate_cfg.max_new_tokens,
                temperature=generate_cfg.temperature,
                top_k=generate_cfg.top_k,
                top_p=generate_cfg.top_p,
                do_sample=generate_cfg.do_sample,
                repetition_penalty=generate_cfg.repetition_penalty,
                no_repeat_ngram_size=generate_cfg.no_repeat_ngram_size,
            )
            
            # Decode batch - extract only new tokens (skip input tokens)
            input_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[:, input_length:]
            batch_captions = tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            batch_captions = [caption.strip() for caption in batch_captions]
            captions.extend(batch_captions)
    
    return captions


class MetricComputer:
    """Computes evaluation metrics for generated captions."""
    
    def __init__(
        self,
        metric_cfgs: Iterable[DictConfig],
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.metric_cfgs = metric_cfgs
        self.metrics = []
        for metric_cfg in metric_cfgs:
            metric = evaluate.load(metric_cfg.name)
            self.metrics.append((metric, metric_cfg))
        self.tokenizer = tokenizer

        self.metrics_results: Dict[str, Any] = {}
        self.predictions: List[str] = []
        self.references: List[str] = []

    def _calculate_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Any]:
        """Calculate metrics comparing predictions to references."""
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

            # Compute average for metrics that return multiple scores
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float, np.floating)):
                        results[f"{metric_cfg.name}_{k}"] = v
                    elif isinstance(v, list):
                        results[f"{metric_cfg.name}_{k}"] = float(np.mean(v))
            else:
                results[metric_cfg.name] = value

            results[metric_cfg.name] = value
        return results

    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute metrics from model evaluation predictions."""
        predictions = eval_pred.predictions
        references = eval_pred.label_ids

        results: Dict[str, Any] = {}
        if self.tokenizer is not None:
            if len(predictions.shape) == 3:
                predictions = np.argmax(predictions, axis=-1)
            
            references = np.where(
                references != -100, references, self.tokenizer.pad_token_id
            )

            decoded_preds = self.tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_labels = self.tokenizer.batch_decode(
                references, skip_special_tokens=True
            )
            results = self._calculate_metrics(decoded_preds, decoded_labels)
            
        return results

    def compute_test_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """Compute metrics from lists of predictions and references."""
        results = self._calculate_metrics(predictions, references)

        self.predictions = self.predictions + predictions
        self.references = self.references + references
        self.metrics_results = results

        return results


    def save_predictions(
        self,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Save predictions and references to CSV file.

        Args:
            output_dir: Directory to save predictions file

        Returns:
            Dictionary with evaluation metrics
        """
        
        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(self.metrics_results, indent=2))
        logger.info(f"Metrics saved to {metrics_path}")

        # Save predictions
        records = [
            {
                "reference": reference,
                "prediction": prediction,
            }
            for reference, prediction in zip(self.references, self.predictions)
        ]
        predictions_path = output_dir / "predictions.csv"
        pd.DataFrame(records).to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

        logger.info(f"Evaluation metrics: {self.metrics_results}")
        return self.metrics_results