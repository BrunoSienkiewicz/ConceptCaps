# VAE Training Module

This module provides a parametrizable PyTorch Lightning implementation for training Variational Autoencoders (VAE) on multi-label tag data.

## Overview

The VAE training system is built with:
- **PyTorch Lightning**: For training loops and distributed training support
- **Hydra**: For configuration management and hyperparameter optimization
- **Multi-label VAE**: For learning latent representations of music tags

## Project Structure

```
src/vae/
├── __init__.py              # Module exports
├── config.py                # Configuration dataclasses
├── modeling.py              # VAE model architecture
├── data.py                  # Data loading and preprocessing
└── lightning_module.py      # PyTorch Lightning training module

src/scripts/vae/
└── run_vae_training.py      # Main training script with Hydra integration

config/
├── vae_training.yaml        # Main config file
├── model/vae/               # Model configurations
│   ├── default.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── large.yaml
├── data/vae/                # Data configurations
│   └── default.yaml
└── trainer/vae/             # Trainer configurations
    ├── default.yaml
    ├── quick_test.yaml
    └── distributed.yaml

scripts/
└── run_vae_training.sh      # Shell script wrapper
```

## Features

### Model Architectures

The module supports multiple VAE architectures through configuration:

- **Standard VAE**: Classic VAE with balanced reconstruction and regularization
- **Small**: 32-dim latent space, 128-dim hidden layer
- **Medium**: 64-dim latent space, 256-dim hidden layer
- **Large**: 128-dim latent space, 512-dim hidden layer (with batch norm)
- **Beta-VAE**: Disentangled VAE with weighted KL divergence
  - **Beta-Default**: β=4.0, balanced disentanglement
  - **Beta-Balanced**: β=4.0, larger networks
  - **Beta-High**: β=10.0, strong disentanglement

### Training Features

- **Multi-label data handling**: Converts tags to one-hot vectors
- **Denoising capability**: Input dropout for robust learning
- **Flexible loss**: Configurable BCE/MSE + KL divergence weights
- **Automatic device detection**: CPU/GPU/Multi-GPU support
- **Logging and checkpointing**: Integration with Weights & Biases (optional)

## Usage

### Basic Training

```bash
# Using default configuration
python src/scripts/vae/run_vae_training.py

# Using shell script
./scripts/run_vae_training.sh
```

### Custom Model and Trainer

```bash
# Small model with default trainer
python src/scripts/vae/run_vae_training.py model=vae/small

# Medium model with distributed training
python src/scripts/vae/run_vae_training.py model=vae/medium trainer=vae/distributed

# Large model for quick testing
python src/scripts/vae/run_vae_training.py model=vae/large trainer=vae/quick_test
```

### Hyperparameter Override

```bash
# Custom epochs and batch size
python src/scripts/vae/run_vae_training.py \
  trainer.max_epochs=300 \
  data.batch_size=64 \
  trainer.optimizer.lr=1e-3

# Custom loss weights
python src/scripts/vae/run_vae_training.py \
  loss.bce_weight=1.0 \
  loss.kld_weight=0.5
```

### Using the Shell Script

```bash
# Default configuration
./scripts/run_vae_training.sh

# Small model with 500 epochs
./scripts/run_vae_training.sh --model small --epochs 500

# Large model with distributed training and custom batch size
./scripts/run_vae_training.sh --model large --trainer distributed --batch-size 64

# Custom learning rate
./scripts/run_vae_training.sh --lr 1e-3

# All options
./scripts/run_vae_training.sh --model medium --trainer default --epochs 250 --batch-size 32 --lr 5e-4 --run-name my_experiment
```

## Configuration

All configurations use YAML files with Hydra. Key configuration sections:

### Model Config (`config/model/vae/*.yaml`)

```yaml
input_dim: 0  # Computed automatically
latent_dim: 32
hidden_dim: 128
dropout_p: 0.3
use_batch_norm: false
```

### Data Config (`config/data/vae/default.yaml`)

```yaml
taxonomy_path: ../data/concepts_to_tags.json
dataset_name: google/MusicCaps
dataset_split: train
aspect_column: aspect_list
batch_size: 32
dataloader_num_workers: 4
shuffle: true
```

### Trainer Config (`config/trainer/vae/*.yaml`)

```yaml
max_epochs: 250
accelerator: auto
devices: auto
strategy: null
precision: '32'
gradient_clip_val: 1.0
enable_progress_bar: true
optimizer:
  lr: 5e-4
  betas: [0.9, 0.999]
  weight_decay: 0.0
```

### Loss Config (`config/vae_training.yaml`)

```yaml
loss:
  bce_weight: 1.0
  kld_weight: 1.0
  use_binary_cross_entropy: true
```

## Output

Training outputs are saved to:

```
outputs/vae_training/
└── {YYYY-MM-DD_HH-MM-SS}/
    ├── lightning_logs/        # Lightning logging
    ├── checkpoints/           # Model checkpoints
    ├── final.ckpt             # Final model checkpoint
    ├── metrics.json           # Training metrics
    ├── summary.json           # Metrics + configuration
    └── .hydra/                # Hydra config snapshots
```

Trained model weights are saved to:

```
models/{model_name}.pth
```

Final metrics are also copied to:

```
models/metrics/
├── {model_name}_metrics.json    # Final metrics
└── {model_name}_summary.json    # Metrics + config
```

## Model Architecture

### VAE Network

The VAE consists of:

**Encoder**:
```
Input (input_dim) 
  → Linear(input_dim, hidden_dim) 
  → ReLU 
  → [Linear(hidden_dim, latent_dim) for μ, σ]
```

**Decoder**:
```
Latent (latent_dim) 
  → Linear(latent_dim, hidden_dim) 
  → ReLU 
  → Linear(hidden_dim, input_dim) 
  → Sigmoid
```

**Loss Function**:
```
L = bce_weight * BCE(recon, x) + kld_weight * KL(q(z|x) || p(z))
```

where:
- BCE: Binary cross-entropy (or MSE) reconstruction loss
- KL: Kullback-Leibler divergence (standard normal prior)

## Extending the Module

### Adding a New Model Architecture

1. Create a new config file: `config/model/vae/custom.yaml`
2. Update the `VAEModelConfig` in `src/vae/config.py` if needed
3. Run with: `python src/scripts/vae/run_vae_training.py model=vae/custom`

### Custom Loss Function

Modify the `_compute_loss` method in `src/vae/lightning_module.py`:

```python
def _compute_loss(self, recon_x, x, mu, logvar):
    # Your custom loss computation
    ...
```

### Adding Callbacks

Add callback configurations to `config/callbacks/` and reference them in `config/vae_training.yaml`:

```yaml
callbacks: custom_callback
```

## Requirements

- PyTorch
- PyTorch Lightning
- Hydra
- Hugging Face Datasets
- NumPy
- Pandas
- scipy

## Notes

- The input dimension is automatically computed from the taxonomy file
- The model uses dropout on inputs during training for denoising
- Train/val/test splits are created automatically (90%/5%/5% by default)
- All configurations can be overridden via command-line arguments
- Checkpoints are saved automatically during training
