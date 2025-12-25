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
from transformers import pipeline
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def generate_caption_tokenized(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Generate a single caption from tokenized inputs."""
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
        )

    input_caption = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
    output_caption = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return input_caption, output_caption


def generate_batch_caption_tokenized(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> List[str]:
    """Generate captions for a batch of tokenized inputs."""
    model.eval()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
        )

    batch_captions = tokenizer.batch_decode(
        outputs, skip_special_tokens=True
    )
    batch_captions = [caption.strip() for caption in batch_captions]
    return batch_captions


def calculate_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = None,
) -> float:
    """
    Calculate perplexity for a given text.
    
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
        shift_labels = input_ids[:, 1:].contiguous()    # Remove first token
        
        # Flatten for loss calculation
        # shift_logits: [batch_size * (seq_len - 1), vocab_size]
        # shift_labels: [batch_size * (seq_len - 1)]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    
    ppl = torch.exp(loss).item()
    return ppl


def evaluate_with_llm_judge(
    captions: List[str],
    prompts: List[str],
    judge_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    judge_template: Optional[str] = None,
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, Any]:
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
    
    for i in tqdm(range(0, len(captions), batch_size), desc="LLM Judge Evaluation"):
        batch_captions = captions[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        
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
            
            # Extract scores and reasoning from batch responses
            for response in responses:
                judge_output = response["generated_text"]
                
                # Remove the input prompt from the output if present
                for judge_prompt in judge_prompts:
                    if judge_output.startswith(judge_prompt):
                        judge_output = judge_output[len(judge_prompt):].strip()
                        break
                
                # Extract score and reasoning
                score = None
                reasoning = ""
                
                for line in judge_output.split('\n'):
                    line = line.strip()
                    if line.startswith("Score:"):
                        try:
                            score_text = line.replace("Score:", "").strip()
                            # Handle formats like "Score: 8/10" or "Score: 8"
                            score_text = score_text.split('/')[0].strip()
                            score = float(score_text.split()[0])
                            # Clamp score to valid range
                            score = max(1.0, min(10.0, score))
                        except:
                            score = 5.0  # Default score if parsing fails
                    elif line.startswith("Reasoning:"):
                        reasoning = line.replace("Reasoning:", "").strip()
                
                scores.append(score if score is not None else 5.0)
                reasonings.append(reasoning if reasoning else "No reasoning provided")
                
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


def generate_captions_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generate_cfg: DictConfig,
    batch_size: int = 8,
    compute_perplexity: bool = False,
    compute_llm_judge: bool = False,
    llm_judge_config: Optional[Dict[str, Any]] = None,
) -> tuple[List[str], Optional[Dict[str, Any]]]:
    """
    Generate captions for a batch of prompts efficiently with optional quality metrics.
    
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
    
    # Extract generation parameters from config
    gen_kwargs = {
        'max_new_tokens': getattr(generate_cfg, 'max_new_tokens', 128),
        'temperature': getattr(generate_cfg, 'temperature', 0.7),
        'top_k': getattr(generate_cfg, 'top_k', 50),
        'top_p': getattr(generate_cfg, 'top_p', 0.9),
        'do_sample': getattr(generate_cfg, 'do_sample', True),
        'repetition_penalty': getattr(generate_cfg, 'repetition_penalty', 1.0),
        'no_repeat_ngram_size': getattr(generate_cfg, 'no_repeat_ngram_size', 0),
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    
    # Add optional parameters if present
    if hasattr(generate_cfg, 'num_beams'):
        gen_kwargs['num_beams'] = generate_cfg.num_beams
    if hasattr(generate_cfg, 'early_stopping'):
        gen_kwargs['early_stopping'] = generate_cfg.early_stopping
    if hasattr(generate_cfg, 'use_cache'):
        gen_kwargs['use_cache'] = generate_cfg.use_cache
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating captions"):
            batch_prompts = prompts[i : i + batch_size]
            
            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(generate_cfg, 'max_length', 512),
            ).to(model.device)
            
            # Generate batch with all configuration parameters
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            
            # Decode batch - handle variable-length prompts
            # Get actual length of each prompt (excluding padding)
            prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
            
            # Extract only generated tokens for each sample
            batch_captions = []
            for j, (output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
                # Extract only the newly generated tokens (after the prompt)
                generated_tokens = output[prompt_len:]
                # Decode only the generated part
                caption = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                batch_captions.append(caption)
            
            captions.extend(batch_captions)

            if compute_perplexity:
                for caption in batch_captions:
                    try:
                        ppl = calculate_perplexity(model, tokenizer, caption, device=model.device)
                        perplexities.append(ppl)
                    except Exception as e:
                        logger.warning(f"Failed to calculate perplexity: {e}")
                        perplexities.append(float('inf'))
            
    
    # Compile quality metrics
    quality_metrics = {}
    
    if compute_perplexity:
        # Filter out infinite values for statistics
        valid_perplexities = [p for p in perplexities if p != float('inf')]
        if valid_perplexities:
            quality_metrics['perplexity'] = {
                'mean': float(np.mean(valid_perplexities)),
                'std': float(np.std(valid_perplexities)),
                'min': float(np.min(valid_perplexities)),
                'max': float(np.max(valid_perplexities)),
                'median': float(np.median(valid_perplexities)),
                'all_scores': perplexities,
            }
            logger.info(f"Perplexity - Mean: {quality_metrics['perplexity']['mean']:.2f}, "
                        f"Std: {quality_metrics['perplexity']['std']:.2f}")
    
    if compute_llm_judge:
        if llm_judge_config is None:
            llm_judge_config = {
                'judge_model_name': 'meta-llama/Llama-3.2-3B-Instruct',
                'batch_size': 4,
                'device': str(model.device),
            }
        
        llm_judge_results = evaluate_with_llm_judge(
            captions=captions,
            prompts=prompts,
            **llm_judge_config,
        )
        quality_metrics['llm_judge'] = llm_judge_results
        logger.info(f"LLM Judge - Mean Score: {llm_judge_results['llm_judge_mean_score']:.2f}, "
                    f"Std: {llm_judge_results['llm_judge_std_score']:.2f}")
    
    return captions, quality_metrics


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

    def _compute_metrics(
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

        return results

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """Compute metrics given model outputs and references."""

        self.predictions.extend(predictions)
        self.references.extend(references)

        self.metrics_results = self._compute_metrics(
            predictions=predictions,
            references=references,
        )
        return self.metrics_results

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

        logger.info(f"Evaluation metrics: {self.metrics_results}")
        
        results = {}
        for key, value in self.metrics_results.items():
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

        # Save predictions
        records = [
            {
                "reference": reference,
                "prediction": prediction,
            }
            for reference, prediction in zip(self.references, self.predictions)
        ]
        predictions_path = output_dir / "test_predictions.csv"
        pd.DataFrame(records).to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

        return self.metrics_results