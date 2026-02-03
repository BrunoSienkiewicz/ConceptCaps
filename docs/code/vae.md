# VAE Module

Beta-VAE for learning plausible music attribute co-occurrence patterns.

This module implements a Variational Autoencoder that learns the joint distribution of music attributes from MusicCaps, enabling sampling of realistic attribute combinations.

## Overview

The VAE learns which attributes naturally co-occur in music (e.g., "jazz" often pairs with "saxophone" and "swing rhythm"). This prevents generating implausible combinations like "heavy metal lullaby".

**Key insight**: Separates semantic modeling (VAE) from text generation (LLM).

## Configuration

::: vae.config
    options:
      show_root_heading: false
      members:
        - VAEConfig
        - VAEModelConfig
        - VAEDataConfig
        - VAELossConfig

## BetaVAE

The core model architecture with disentanglement via β-weighted KL divergence.

::: vae.model.BetaVAE
    options:
      show_root_heading: true
      members:
        - __init__
        - encode
        - decode
        - reparameterize
        - forward
        - sample
        - reconstruct

## BetaVAELightningModule

PyTorch Lightning wrapper for training.

::: vae.lightning_module.BetaVAELightningModule
    options:
      show_root_heading: true
      members:
        - __init__
        - forward
        - training_step
        - validation_step
        - configure_optimizers

## VAEDataModule

Data loading for multi-label tag data.

::: vae.data.VAEDataModule
    options:
      show_root_heading: true
      members:
        - __init__
        - setup
        - train_dataloader
        - val_dataloader
        - test_dataloader

## Evaluation

::: vae.evaluation
    options:
      show_root_heading: false
      members:
        - VAEMetrics
        - MetricsSaver
        - MetricsSaveCallback
