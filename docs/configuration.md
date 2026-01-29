# Configuration Reference

This document describes all available configuration options for ConceptCaps.

## Main Configuration Files

### VAE Training (`config/vae_training.yaml`)

Configuration for training the Variational Autoencoder.

```yaml
defaults:
  - model: vae/default
  - data: vae/default
  - trainer: vae/default
  - callbacks: default
  - logger: wandb
  - paths: default
  - hydra: default
```

### Caption Fine-tuning (`config/caption_fine_tuning.yaml`)

Configuration for fine-tuning the caption generation model.

```yaml
defaults:
  - model: caption/llama
  - data: caption/default
  - trainer: caption/default
  - lora: caption/default
  - callbacks: default
  - logger: wandb
  - paths: default
  - prompt: llama
  - hydra: default
```

### Caption Inference (`config/caption_inference.yaml`)

Configuration for generating captions from attribute lists.

```yaml
defaults:
  - model: caption/llama
  - data: caption/inference
  - generation: caption/default
  - prompt: llama
  - paths: default
  - hydra: default
```

### TTA Inference (`config/tta_inference.yaml`)

Configuration for text-to-audio generation.

```yaml
defaults:
  - model: tta/musicgen
  - data: tta/default
  - generation: tta/default
  - evaluation: tta/default
  - paths: default
  - hydra: default
```

## Configuration Groups

### Callbacks (`config/callbacks/`)

| File | Description |
|------|-------------|
| `default.yaml` | Standard callback set |
| `early_stopping.yaml` | Early stopping configuration |
| `model_checkpoint.yaml` | Checkpoint saving settings |
| `model_summary.yaml` | Model summary logging |
| `none.yaml` | No callbacks |
| `rich_progress_bar.yaml` | Rich progress bar |

### Data (`config/data/`)

#### Caption Data (`config/data/caption/`)

```yaml
# Example configuration
_target_: src.caption.lightning_datamodule.CaptionDataModule
train_path: ${paths.data_dir}/train.csv
val_path: ${paths.data_dir}/val.csv
test_path: ${paths.data_dir}/test.csv
batch_size: 8
num_workers: 4
max_length: 512
```

#### VAE Data (`config/data/vae/`)

```yaml
# Example configuration
_target_: src.vae.data.VAEDataModule
data_path: ${paths.data_dir}/tags.csv
batch_size: 32
num_workers: 4
train_split: 0.8
val_split: 0.1
```

#### TTA Data (`config/data/tta/`)

```yaml
# Example configuration
_target_: src.tta.data.TTADataset
captions_path: ${paths.data_dir}/captions.csv
audio_dir: ${paths.data_dir}/audio
sample_rate: 32000
duration: 10
```

### Models (`config/model/`)

#### Caption Models (`config/model/caption/`)

```yaml
# LLaMA configuration
_target_: src.caption.model.CaptionModel
base_model: "meta-llama/Llama-2-7b-hf"
load_in_8bit: true
device_map: auto
```

#### VAE Models (`config/model/vae/`)

```yaml
# VAE configuration
_target_: src.vae.model.VAE
input_dim: 200
latent_dim: 32
hidden_dims: [128, 64]
dropout: 0.1
```

#### TTA Models (`config/model/tta/`)

```yaml
# MusicGen configuration
_target_: src.tta.audio.MusicGenWrapper
model_name: "facebook/musicgen-medium"
device: cuda
```

### Trainer (`config/trainer/`)

#### VAE Trainer (`config/trainer/vae/`)

```yaml
_target_: lightning.Trainer
max_epochs: 100
accelerator: gpu
devices: 1
precision: 16-mixed
gradient_clip_val: 1.0
```

#### Caption Trainer (`config/trainer/caption/`)

```yaml
_target_: lightning.Trainer
max_epochs: 5
accelerator: gpu
devices: 1
precision: bf16-mixed
accumulate_grad_batches: 4
```

### LoRA (`config/lora/caption/`)

```yaml
# LoRA configuration for caption fine-tuning
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
bias: none
task_type: CAUSAL_LM
```

### Logger (`config/logger/`)

#### Weights & Biases (`config/logger/wandb.yaml`)

```yaml
_target_: lightning.pytorch.loggers.WandbLogger
project: conceptcaps
name: ${hydra:runtime.choices.experiment}
save_dir: ${paths.output_dir}
log_model: false
tags: []
```

### Paths (`config/paths/`)

#### Default (`config/paths/default.yaml`)

```yaml
root_dir: ${oc.env:PROJECT_ROOT}
data_dir: ${paths.root_dir}/data
output_dir: ${paths.root_dir}/outputs
model_dir: ${paths.root_dir}/models
log_dir: ${paths.output_dir}/logs
```

#### PLGrid (`config/paths/plgrid.yaml`)

```yaml
# HPC cluster paths
root_dir: /net/tscratch/people/plg<username>/conceptcaps
data_dir: ${paths.root_dir}/data
output_dir: ${paths.root_dir}/outputs
model_dir: ${paths.root_dir}/models
```

### Prompt Templates (`config/prompt/`)

All prompt templates follow this structure:

```yaml
system_prompt: |
  You are a professional music critic...

user_template: |
  Generate a detailed caption for music with these attributes:
  {attributes}

assistant_prefix: |
  This
```

Available templates:
- `default.yaml`: General-purpose template
- `llama.yaml`: Optimized for LLaMA models
- `llama_short.yaml`: Shorter outputs
- `llama_zero_shot.yaml`: Zero-shot prompting
- `mistral.yaml`: Mistral-optimized
- `qwen.yaml`: Qwen-optimized
- `gpt.yaml`: GPT-style formatting

### Evaluation (`config/evaluation/`)

#### Caption Evaluation (`config/evaluation/caption/`)

```yaml
metrics:
  - bertscore
  - rouge
  - mauve

bertscore:
  model_type: microsoft/deberta-xlarge-mnli
  lang: en

rouge:
  rouge_types: [rouge1, rouge2, rougeL]
```

#### TTA Evaluation (`config/evaluation/tta/`)

```yaml
metrics:
  - clap_score

clap:
  model_name: laion/clap-htsat-unfused
  device: cuda
```

### Generation (`config/generation/`)

#### Caption Generation (`config/generation/caption/`)

```yaml
max_new_tokens: 256
temperature: 0.7
top_p: 0.9
top_k: 50
do_sample: true
repetition_penalty: 1.1
```

#### TTA Generation (`config/generation/tta/`)

```yaml
duration: 10
guidance_scale: 3.0
max_cfg_coef: 10.0
min_cfg_coef: 1.0
```

### Sweeps (`config/sweeps/`)

Hyperparameter sweep configurations for Weights & Biases.

#### VAE Sweep (`config/sweeps/vae_wandb_sweep.yaml`)

```yaml
program: src/scripts/vae_training.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  model.latent_dim:
    values: [16, 32, 64, 128]
  model.hidden_dims:
    values: [[64, 32], [128, 64], [256, 128]]
  trainer.max_epochs:
    value: 100
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_ROOT` | Project root directory | Auto-detected |
| `WANDB_API_KEY` | W&B API key | None |
| `HF_TOKEN` | Hugging Face token | None |
| `CUDA_VISIBLE_DEVICES` | GPU selection | All |

## Command Line Overrides

### Basic Overrides

```bash
# Single parameter
python -m src.scripts.vae_training model.latent_dim=64

# Multiple parameters
python -m src.scripts.vae_training model.latent_dim=64 trainer.max_epochs=50

# Nested parameters
python -m src.scripts.caption_fine_tuning lora.r=32 lora.lora_alpha=64
```

### Config Group Selection

```bash
# Select specific config from group
python -m src.scripts.caption_fine_tuning model=caption/llama prompt=llama_short

# Use preset
python -m src.scripts.caption_fine_tuning preset=caption/default
```

### Multirun

```bash
# Run with multiple values
python -m src.scripts.vae_training --multirun model.latent_dim=16,32,64,128
```
