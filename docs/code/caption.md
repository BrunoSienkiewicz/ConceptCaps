# Caption Module

LLM fine-tuning for music caption generation from concept tags.

This module handles the conversion of structured music attributes (tags) into natural language descriptions using fine-tuned language models with LoRA adapters.

## Overview

The caption pipeline:

1. **Data preparation**: Format prompts with attribute lists
2. **Fine-tuning**: Train LLM with LoRA on MusicCaps-derived data  
3. **Inference**: Generate captions from new attribute combinations

## CaptionFineTuningModule

::: caption.lightning_module.CaptionFineTuningModule
    options:
      show_root_heading: true
      members:
        - __init__
        - forward
        - training_step
        - validation_step
        - configure_optimizers

## CaptionDataModule

::: caption.lightning_datamodule.CaptionDataModule
    options:
      show_root_heading: true

## Data Processing

::: caption.data
    options:
      show_root_heading: false
      members:
        - prepare_datasets
        - prepare_inference_datasets

## Model Utilities

::: caption.model
    options:
      show_root_heading: false
      members:
        - prepare_tokenizer
        - prepare_training_model
        - build_quantization_config

## Evaluation

::: caption.evaluation
    options:
      show_root_heading: false
      members:
        - calculate_perplexity
        - evaluate_with_llm_judge
        - MetricComputer

## Inference

::: caption.inference
    options:
      show_root_heading: false
      members:
        - generate_captions_batch
