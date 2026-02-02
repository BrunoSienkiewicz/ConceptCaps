# TCAV Module

Testing with Concept Activation Vectors for model interpretability.

This module implements TCAV analysis to quantify how much a classifier relies on human-defined concepts (genres, moods, instruments) when making predictions.

## Overview

TCAV workflow:

1. **Extract activations** from the classifier's bottleneck layer
2. **Train CAVs** (linear classifiers) to separate concept vs. random examples
3. **Compute directional derivatives** to measure concept influence on predictions

## TCAV

Main class for concept-based interpretability analysis.

::: tcav.tcav.TCAV
    options:
      show_root_heading: true
      members:
        - __init__
        - get_activations
        - train_cav
        - get_directional_derivatives
        - compute_tcav_score

## MusicGenreClassifier

CNN-based classifier with exposed bottleneck layer for TCAV analysis.

::: tcav.model.MusicGenreClassifier
    options:
      show_root_heading: true
      members:
        - __init__
        - forward
        - training_step
        - validation_step
        - configure_optimizers
