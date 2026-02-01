<div align="center">

# ConceptCaps

### A Distilled Concept Dataset for Interpretability in Music Models

[![arXiv](https://img.shields.io/badge/arXiv-2601.14157-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.14157)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ConceptCaps-yellow.svg?style=flat-square)](https://huggingface.co/datasets/bsienkiewicz/ConceptCaps)
[![License](https://img.shields.io/badge/License-CC--BY--4.0-green.svg?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

</div>

## Overview

ConceptCaps is a music captioning dataset derived from [MusicCaps](https://huggingface.co/datasets/google/MusicCaps), specifically designed for concept-based interpretability research in text-to-audio (TTA) generation systems.
The dataset provides categorized musical concept annotations from distilled taxonomy (200 unique tags) alongside natural language captions, enabling fine-grained analysis of how TTA models represent and generate musical concepts.

## TL;DR

Concept-based interpretability methods like TCAV require clean, well-separated positive and negative examples for each concept. Existing music datasets lack this structure. ConceptCaps addresses this by:

1. Using a **VAE** to learn plausible attribute co-occurrence patterns
2. **Fine-tuning an LLM** to convert attribute lists into professional descriptions
3. Synthesizing audio with **MusicGen**

This pipeline resulted in large high-quality dataset that can be successfuly used for interpretability research, as demonstrated by downstream TCAV analysis.

### Generation Pipeline

![](./docs/assets/pipeline.pdf)

### Key Features

- **21k music-caption-audio triplets** with explicit labels from a 200-attribute taxonomy
- **178 hours of audio content** paired with textual descriptions
- **Four concept categories**: genre, mood, instruments, tempo
- **Separated semantic modeling from text generation**: VAE learns attribute co-occurrence, LLM generates descriptions
- **Validated through multiple metrics**: CLAP alignment, BERTScore, MAUVE, and TCAV analysis

This separation improves coherence and controllability over end-to-end approaches.

## Dataset

The dataset is available on Hugging Face: **[bsienkiewicz/ConceptCaps](https://huggingface.co/datasets/bsienkiewicz/ConceptCaps)**

### Quick Start

```python
from datasets import load_dataset

# Load captions only
dataset = load_dataset("bsienkiewicz/ConceptCaps", "default")

# Load with audio
dataset = load_dataset("bsienkiewicz/ConceptCaps", "audio")
```

## Project Structure

```
├── config/                 # Hydra configuration files
│   ├── callbacks/          # Training callbacks (checkpoints, early stopping, etc.)
│   ├── data/               # Data module configurations
│   ├── evaluation/         # Evaluation metric configurations
│   ├── generation/         # Generation pipeline configs
│   ├── logger/             # Logging configurations
│   ├── lora/               # LoRA fine-tuning configs
│   ├── model/              # Model architecture configs
│   ├── paths/              # Path configurations
│   ├── preset/             # Preset configurations
│   ├── prompt/             # LLM prompt templates
│   ├── sweeps/             # Hyperparameter sweep configs
│   └── trainer/            # PyTorch Lightning trainer configs
│
├── data/                   # Datasets and intermediate data
│
├── docs/                   # Documentation
│   ├── DATASET_CARD.md     # Dataset card for Hugging Face
│   ├── experiments/        # Experiment documentation
│   └── assets/             # Documentation assets
│
├── models/                 # Trained model checkpoints
│   ├── best-genre-classifier.ckpt      # Genre classifier for TCAV
│   └── vae_final.pth                   # Final VAE model
│
├── notebooks/              # Jupyter notebooks for analysis
│
├── scripts/                # Executable scripts
│   ├── caption/            # Caption generation scripts
│   ├── helper/             # Utility scripts
│   ├── tta/                # Text-to-audio scripts
│   └── vae/                # VAE training scripts
│
├── src/                    # Source code
│   ├── caption/            # Caption generation module
│   ├── vae/                # Variational Autoencoder module
│   ├── tcav/               # TCAV analysis module
│   ├── tta/                # Text-to-audio module
│   ├── utils/              # Shared utilities
│   └── constants.py        # Project constants
│
├── outputs/                # Training outputs and experiments
├── environment.yml         # Conda environment specification
├── Makefile                # Common development commands
└── mkdocs.yml              # Documentation configuration
```

Every module in `src` directory follows roughly this file structure:

```
├──<module>/
│   ├── data.py         # data processing
│   ├── model.py        # model architecture
│   ├── inference.py    # sampling/generation
│   └── evaluation.py   # evaluation metrics
```

## Installation

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended)
- Conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/BrunoSienkiewicz/ConceptCaps
cd ConceptCaps

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate conceptcaps
```

## Usage

### Run Script with Default Parameters

```bash
python -m src.scripts.vae.train
```

### Configuration Override

Override any parameter from command line:

```bash
python -m src.scripts.vae.train trainer.max_epochs=100 data.batch_size=64
```

### Run Configuration Preset

```bash
python -m src.scripts.vae.train +preset=vae/default
```

## Notebooks

The repository includes Jupyter notebooks demonstrating each pipeline stage:

| Notebook | Description |
|----------|-------------|
| `1. Taxonomy and dataset distillation.ipynb` | Methodology in taxonomy creation and tag mapping |
| `2. VAE aspect modeling.ipynb` | VAE attribute sampling |
| `3. MusicCaps and VAE generated dataset comparison.ipynb` | Dataset quality analysis |
| `4. Conditioned caption inference.ipynb` | Caption generation from concepts |
| `5. Audio generation analysis.ipynb` | MusicGen audio synthesis |
| `6. Create final datasets.ipynb` | Final dataset preparation |
| `7. TCAV for genre classification.ipynb` | TCAV interpretability analysis |

## License

This project is licensed under the [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements

We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2025/018397.

## Citation

If you use ConceptCaps in your research, please cite:

```bibtex
@article{sienkiewicz2026conceptcaps,
  title={ConceptCaps -- a Distilled Concept Dataset for Interpretability in Music Models},
  author={Sienkiewicz, Bruno and Neumann, Łukasz and Modrzejewski, Mateusz},
  journal={arXiv preprint arXiv:2601.14157},
  year={2026}
}
```

## Authors

- [Bruno Sienkiewicz](https://arxiv.org/search/cs?searchtype=author&query=Sienkiewicz,+B)
- [Łukasz Neumann](https://arxiv.org/search/cs?searchtype=author&query=Neumann,+%C5%81)
- [Mateusz Modrzejewski](https://arxiv.org/search/cs?searchtype=author&query=Modrzejewski,+M)