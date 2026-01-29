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

This repository contains the source code, configurations, and data used to create the **ConceptCaps** dataset — a concept-based music captioning dataset designed for interpretability research in text-to-audio (TTA) generation systems.

ConceptCaps provides structured musical concept annotations alongside natural language captions, enabling fine-grained analysis of how TTA models represent and generate musical concepts.

### Key Features

- **23k music-caption-audio triplets** with explicit labels from a 200-attribute taxonomy
- **Four concept categories**: genre, mood, instruments, tempo
- **Separated semantic modeling from text generation**: VAE learns attribute co-occurrence, LLM generates descriptions
- **Validated through multiple metrics**: CLAP alignment, BERTScore, MAUVE, and TCAV analysis

## TL;DR

Concept-based interpretability methods like TCAV require clean, well-separated positive and negative examples for each concept. Existing music datasets lack this structure. ConceptCaps addresses this by:

1. Using a **VAE** to learn plausible attribute co-occurrence patterns
2. **Fine-tuning an LLM** to convert attribute lists into professional descriptions
3. Synthesizing audio with **MusicGen**

This separation improves coherence and controllability over end-to-end approaches.

## Dataset

The dataset is available on Hugging Face: **[bsienkiewicz/ConceptCaps](https://huggingface.co/datasets/bsienkiewicz/ConceptCaps)**

### Configurations

| Configuration | Samples | Audio |
|--------------|---------|-------|
| `default` | 5,358 | ❌ |
| `25pct` | 1,339 | ❌ |
| `10pct` | 535 | ❌ |
| `audio` | 5,358 | ✅ |
| `25pct-audio` | 1,339 | ✅ |
| `10pct-audio` | 535 | ✅ |

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
│   │   ├── caption/        # Caption dataset configs
│   │   ├── tta/            # Text-to-audio configs
│   │   └── vae/            # VAE dataset configs
│   ├── evaluation/         # Evaluation metric configurations
│   ├── generation/         # Generation pipeline configs
│   ├── logger/             # Logging configurations (W&B)
│   ├── lora/               # LoRA fine-tuning configs
│   ├── model/              # Model architecture configs
│   ├── paths/              # Path configurations
│   ├── preset/             # Preset configurations
│   ├── prompt/             # LLM prompt templates
│   ├── sweeps/             # Hyperparameter sweep configs
│   └── trainer/            # PyTorch Lightning trainer configs
│
├── data/                   # Datasets and intermediate data
│   ├── concepts_to_tags.json           # Concept taxonomy mapping
│   ├── musiccaps_tag_frequencies.csv   # Tag frequency analysis
│   ├── evaluation_results/             # Evaluation outputs
│   ├── generated_captions/             # Generated caption datasets
│   └── mtg_jamendo/                    # MTG-Jamendo data
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
│   ├── 1. Taxonomy and dataset distillation.ipynb
│   ├── 2. VAE aspect modeling.ipynb
│   ├── 3. MusicCaps and VAE generated dataset comparison.ipynb
│   ├── 4. Conditioned caption inference.ipynb
│   ├── 5. Audio generation analysis.ipynb
│   ├── 6. Create final datasets.ipynb
│   └── 7. TCAV for genre classification.ipynb
│
├── scripts/                # Executable scripts
│   ├── caption/            # Caption generation scripts
│   ├── helper/             # Utility scripts
│   ├── tta/                # Text-to-audio scripts
│   └── vae/                # VAE training scripts
│
├── src/                    # Source code
│   ├── caption/            # Caption generation module
│   │   ├── data.py         # Data loading and processing
│   │   ├── model.py        # Model definitions
│   │   ├── inference.py    # Inference pipeline
│   │   ├── evaluation.py   # Caption evaluation metrics
│   │   └── lightning_*.py  # PyTorch Lightning components
│   │
│   ├── vae/                # Variational Autoencoder module
│   │   ├── data.py         # VAE data processing
│   │   ├── model.py        # VAE architecture
│   │   ├── inference.py    # VAE sampling/generation
│   │   ├── evaluation.py   # VAE evaluation
│   │   └── lightning_module.py
│   │
│   ├── tcav/               # TCAV analysis module
│   │   ├── model.py        # Classifier for TCAV
│   │   └── tcav.py         # TCAV implementation
│   │
│   ├── tta/                # Text-to-audio module
│   │   ├── audio.py        # Audio processing utilities
│   │   ├── data.py         # TTA data handling
│   │   └── evaluation.py   # Audio evaluation metrics
│   │
│   ├── data/               # Common data utilities
│   ├── utils/              # Shared utilities
│   └── constants.py        # Project constants
│
├── outputs/                # Training outputs and experiments
├── environment.yml         # Conda environment specification
├── Makefile                # Common development commands
└── mkdocs.yml              # Documentation configuration
```

## Installation

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended)
- Conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/bsienkiewicz/music-gen-interpretability
cd music-gen-interpretability

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate conceptcaps
```

## Usage

### VAE Training

Train the VAE for learning attribute co-occurrence patterns:

```bash
python -m src.scripts.vae_training
```

### Caption Generation

Generate captions from attribute lists using the fine-tuned LLM:

```bash
python -m src.scripts.caption_inference
```

### Caption Model Fine-tuning

Fine-tune the caption generation model:

```bash
python -m src.scripts.caption_fine_tuning
```

### Text-to-Audio Inference

Generate audio from captions using MusicGen:

```bash
python -m src.scripts.tta_inference
```

### Configuration Override

Override any parameter from command line:

```bash
python -m src.scripts.vae_training trainer.max_epochs=100 data.batch_size=64
```

## Notebooks

The repository includes Jupyter notebooks demonstrating each pipeline stage:

| Notebook | Description |
|----------|-------------|
| `1. Taxonomy and dataset distillation.ipynb` | Concept taxonomy creation and tag mapping |
| `2. VAE aspect modeling.ipynb` | VAE training and attribute sampling |
| `3. MusicCaps and VAE generated dataset comparison.ipynb` | Dataset quality analysis |
| `4. Conditioned caption inference.ipynb` | Caption generation from concepts |
| `5. Audio generation analysis.ipynb` | MusicGen audio synthesis |
| `6. Create final datasets.ipynb` | Final dataset preparation |
| `7. TCAV for genre classification.ipynb` | TCAV interpretability analysis |

## Evaluation Metrics

ConceptCaps is validated through:

- **Audio-Text Alignment**: CLAP scores
- **Linguistic Quality**: BERTScore, MAUVE
- **Interpretability**: TCAV analysis confirming concept probes recover musically meaningful patterns

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

- [Bruno Sienkiewicz](https://github.com/bsienkiewicz)
- [Łukasz Neumann](https://arxiv.org/search/cs?searchtype=author&query=Neumann,+%C5%81)
- [Mateusz Modrzejewski](https://arxiv.org/search/cs?searchtype=author&query=Modrzejewski,+M)