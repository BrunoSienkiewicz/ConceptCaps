# TTA Module

Text-to-audio generation using MusicGen.

This module handles audio synthesis from generated captions using Facebook's MusicGen model, with support for multi-GPU inference.

## Overview

After generating captions from the LLM, this module:

1. Tokenizes captions for MusicGen's text encoder
2. Generates audio in batches (with GPU memory optimization)
3. Saves audio files and computes quality metrics (CLAP, FAD)

## Configuration

::: tta.config
    options:
      show_root_heading: false
      members:
        - TTAConfig

## Audio Generation

### Single-GPU Generation

::: tta.audio.generate_audio_samples
    options:
      show_root_heading: true

### Multi-GPU Generation (Accelerate)

::: tta.audio.generate_audio_samples_accelerate
    options:
      show_root_heading: true

## Data Utilities

::: tta.data
    options:
      show_root_heading: false
      members:
        - prepare_dataloader

## Evaluation

::: tta.evaluation
    options:
      show_root_heading: false
      members:
        - TTAEvaluator

