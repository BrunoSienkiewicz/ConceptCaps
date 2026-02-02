# Configuration

Hydra configuration files for all pipeline stages.

## Overview

The project uses [Hydra](https://hydra.cc/) for hierarchical configuration management. Each training/inference script has a main config that composes modular sub-configs.

## Main Configuration Files

| File | Purpose |
|------|---------|
| `vae_training.yaml` | VAE training for attribute modeling |
| `caption_fine_tuning.yaml` | LLM LoRA fine-tuning |
| `caption_inference.yaml` | Caption generation |
| `tta_inference.yaml` | Audio synthesis with MusicGen |

## Configuration Structure

```
config/
├── callbacks/          # Training callbacks (checkpoints, early stopping)
├── data/               # Dataset configurations
│   ├── caption/        # Caption dataset configs
│   ├── tta/            # Audio generation configs
│   └── vae/            # VAE training data configs
├── evaluation/         # Metric configurations
├── generation/         # Generation parameters
├── logger/             # Logging (W&B) configs
├── lora/               # LoRA adapter configs
├── model/              # Model architecture configs
│   ├── caption/        # LLM model configs
│   ├── tta/            # MusicGen configs
│   └── vae/            # VAE architecture configs
├── paths/              # Path configurations
├── preset/             # Experiment presets
├── prompt/             # LLM prompt templates
├── sweeps/             # Hyperparameter sweep configs
└── trainer/            # PyTorch Lightning trainer configs
```

## Usage

### Override parameters from CLI

```bash
python -m src.scripts.vae_training trainer.max_epochs=100 data.batch_size=64
```

### Use a preset

```bash
python -m src.scripts.caption_fine_tuning +preset=caption/quick_test
```

## Key Configuration Options

### VAE Training

```yaml
# vae_training.yaml
model:
  latent_dim: 32       # Latent space dimension
  hidden_dim: 128      # Hidden layer size
  beta: 1.0            # KL divergence weight

loss:
  bce_weight: 1.0      # Reconstruction loss weight
  kld_weight: 1.0      # KL divergence weight

inference:
  num_samples: 5000    # Samples to generate
  temperature: 1.0     # Sampling temperature
  threshold: 0.5       # Tag activation threshold
```

### Caption Fine-tuning

```yaml
# caption_fine_tuning.yaml
model:
  name: meta-llama/Llama-3.1-8B-Instruct
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: nf4

lora:
  r: 16                # LoRA rank
  lora_alpha: 32       # LoRA alpha
  target_modules:      # Modules to adapt
    - q_proj
    - v_proj
```

### TTA Inference

```yaml
# tta_inference.yaml
model:
  name: facebook/musicgen-small
  
generation:
  max_new_tokens: 512  # ~10 seconds of audio
  guidance_scale: 3.0  # Classifier-free guidance
  temperature: 1.0
```
