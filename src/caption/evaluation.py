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
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction, pipeline

from src.utils import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def calculate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = None,
) -> float:
    """Calculate perplexity for a given text.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Text to calculate perplexity for
        device: Device to use (defaults to model's device)

    Returns:
        Perplexity score (lower is better)
    """
    if device is None:
        device = model.device

    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    with torch.no_grad():
        attention_mask = torch.ones_like(input_ids).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        # logits: [batch_size, seq_len, vocab_size]
        # We predict token i+1 from tokens 0...i
        shift_logits = logits[:, :-1, :].contiguous()  # Remove last prediction
        shift_labels = input_ids[:, 1:].contiguous()  # Remove first token

        # Flatten for loss calculation
        # shift_logits: [batch_size * (seq_len - 1), vocab_size]
        # shift_labels: [batch_size * (seq_len - 1)]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    ppl = torch.exp(loss).item()
    return ppl


def evaluate_with_llm_judge(
    captions: list[str],
    prompts: list[str],
    judge_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    judge_template: str | None = None,
    batch_size: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    """
    Evaluate generated captions using an LLM as a judge.
    Based on: https://huggingface.co/learn/cookbook/llm_judge

    Args:
        captions: Generated captions to evaluate
        prompts: Original prompts used for generation
        judge_model_name: Name of the judge model to use
        batch_size: Batch size for judge evaluation
        device: Device to use

    Returns:
        Dictionary with evaluation scores and reasoning
    """

    # Initialize judge model
    judge_pipeline = pipeline(
        "text-generation",
        model=judge_model_name,
        device=device,
        torch_dtype=torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32,
    )

    if judge_template is None:
        judge_template = """You are an expert music critic evaluating AI-generated music descriptions.

    Given the following tags and the generated description, evaluate the description on a scale of 1-10 based on:
    - Accuracy: Does it correctly incorporate the tags?
    - Coherence: Is it well-written and coherent?
    - Completeness: Does it provide sufficient detail?
    - Musicality: Does it sound like a natural music description?

    Tags/Prompt: {prompt}
    Generated Description: {caption}

    Provide your evaluation in the following format:
    Score: [1-10]
    Reasoning: [Your explanation]
    """

    scores = []
    reasonings = []

    logger.info(f"Evaluating {len(captions)} captions with LLM judge...")

    for i in tqdm(
        range(0, len(captions), batch_size), desc="LLM Judge Evaluation"
    ):
        batch_captions = captions[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        # Prepare batch of judge prompts
        judge_prompts = [
            judge_template.format(prompt=prompt, caption=caption)
            for prompt, caption in zip(batch_prompts, batch_captions)
        ]

        try:
            # Process entire batch at once
            responses = judge_pipeline(
                judge_prompts,
                max_new_tokens=256,
                temperature=0.1,  # Lower temperature for more consistent scoring
                do_sample=False,  # Use greedy decoding for scoring consistency
                pad_token_id=judge_pipeline.tokenizer.eos_token_id,
                batch_size=batch_size,
            )

            # Handle batch response structure - pipeline returns list of lists
            # Each element is a list containing a dict with 'generated_text'
            for response_list in responses:
                # Get the first (and only) generation from each prompt
                response = (
                    response_list[0]
                    if isinstance(response_list, list)
                    else response_list
                )
                judge_output = response["generated_text"]

                # Remove the input prompt from the output if present
                for judge_prompt in judge_prompts:
                    if judge_output.startswith(judge_prompt):
                        judge_output = judge_output[len(judge_prompt) :].strip()
                        break

                # Extract score and reasoning
                score = None
                reasoning = ""

                for line in judge_output.split("\n"):
                    line = line.strip()
                    if line.startswith("Score:"):
                        try:
                            score_text = line.replace("Score:", "").strip()
                            # Handle formats like "Score: 8/10" or "Score: 8"
                            score_text = score_text.split("/")[0].strip()
                            score = float(score_text.split()[0])
                            # Clamp score to valid range
                            score = max(1.0, min(10.0, score))
                        except IndexError:
                            score = 5.0  # Default score if parsing fails
                    elif line.startswith("Reasoning:"):
                        reasoning = line.replace("Reasoning:", "").strip()

                scores.append(score if score is not None else 5.0)
                reasonings.append(
                    reasoning if reasoning else "No reasoning provided"
                )

        except Exception as e:
            logger.warning(f"Failed to evaluate batch with LLM judge: {e}")
            # Add default scores for failed batch
            for _ in range(len(batch_captions)):
                scores.append(5.0)
                reasonings.append("Evaluation failed")

    return {
        "llm_judge_mean_score": float(np.mean(scores)) if scores else 0.0,
        "llm_judge_std_score": float(np.std(scores)) if scores else 0.0,
        "llm_judge_min_score": float(np.min(scores)) if scores else 0.0,
        "llm_judge_max_score": float(np.max(scores)) if scores else 0.0,
        "llm_judge_scores": scores,
        "llm_judge_reasonings": reasonings,
    }


class MetricComputer:
    """Computes evaluation metrics for generated captions."""

    def __init__(
        self,
        metric_cfgs: Iterable[DictConfig],
        tokenizer: AutoTokenizer | None = None,
    ):
        self.metric_cfgs = metric_cfgs
        self.metrics = []
        for metric_cfg in metric_cfgs:
            metric = evaluate.load(metric_cfg.name)
            self.metrics.append((metric, metric_cfg))
        self.tokenizer = tokenizer

        self.metrics_results: dict[str, Any] = {}
        self.predictions: list[str] = []
        self.references: list[str] = []

    def _compute_metrics(
        self, predictions: list[str], references: list[str]
    ) -> dict[str, Any]:
        """Calculate metrics comparing predictions to references."""
        results: dict[str, Any] = {}
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

        return results

    def compute_metrics(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, Any]:
        """Compute metrics given model outputs and references."""

        self.predictions.extend(predictions)
        self.references.extend(references)

        self.metrics_results = self._compute_metrics(
            predictions=predictions,
            references=references,
        )
        return self.metrics_results

    def save_predictions(self, output_dir: Path) -> dict[str, Any]:
        """Save predictions and metrics to output directory.

        Files saved:
        - <output_dir>/evaluation_metrics.json
        - <output_dir>/all_predictions.csv

        Args:
            output_dir: Directory to save predictions file

        Returns:
            Dictionary with evaluation metrics
        """

        logger.info(f"Evaluation metrics: {self.metrics_results}")

        results = {}
        for key, value in self.metrics_results.items():
            # Flatten nested dicts if any
            # Only consider one level of nesting
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (int, float, np.floating)):
                        results[key + "_" + k] = v
            elif isinstance(value, (int, float, np.floating)):
                results[key] = value

        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(results, indent=4))
        logger.info(f"Metrics saved to {metrics_path}")

        # Save predictions with references
        records = [
            {
                "reference": reference,
                "prediction": prediction,
            }
            for reference, prediction in zip(self.references, self.predictions)
        ]
        predictions_path = output_dir / "all_predictions.csv"
        pd.DataFrame(records).to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

        return self.metrics_results
