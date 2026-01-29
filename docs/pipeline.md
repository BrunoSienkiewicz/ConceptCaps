# ConceptCaps Pipeline

This document provides a detailed overview of the ConceptCaps data generation pipeline.

## Pipeline Overview

The ConceptCaps pipeline consists of four main stages:

```
MusicCaps Tags → VAE Sampling → LLM Caption Generation → MusicGen Audio Synthesis
```

### Stage 1: Taxonomy and Concept Extraction

**Notebook**: `1. Taxonomy and dataset distillation.ipynb`

The first stage creates a structured taxonomy from the unstructured MusicCaps `aspect_list` annotations.

#### Process

1. **Tag Collection**: Extract all unique tags from MusicCaps dataset
2. **Frequency Analysis**: Analyze tag frequencies (see `data/musiccaps_tag_frequencies.csv`)
3. **Category Mapping**: Map tags to four categories:
   - **Genre**: Musical style (e.g., jazz, rock, classical)
   - **Mood**: Emotional qualities (e.g., mellow, energetic, sad)
   - **Instruments**: Sound sources (e.g., piano, guitar, drums)
   - **Tempo**: Speed indicators (e.g., slow, fast, moderate)
4. **Taxonomy Creation**: Output stored in `data/concepts_to_tags.json`

#### Output Format

```json
{
  "genre": ["jazz", "rock", "pop", "classical", ...],
  "mood": ["happy", "sad", "energetic", "calm", ...],
  "instrument": ["piano", "guitar", "drums", "violin", ...],
  "tempo": ["slow", "fast", "medium", "uptempo", ...]
}
```

### Stage 2: VAE Aspect Modeling

**Notebook**: `2. VAE aspect modeling.ipynb`  
**Module**: `src/vae/`  
**Config**: `config/vae_training.yaml`

The VAE learns plausible attribute co-occurrence patterns from the taxonomy.

#### Architecture

The VAE model (`src/vae/model.py`) uses:
- **Encoder**: Maps one-hot tag vectors to latent space
- **Latent Space**: Learns meaningful attribute correlations
- **Decoder**: Reconstructs plausible tag combinations

#### Training

```bash
python -m src.scripts.vae_training
```

Configuration options in `config/vae_training.yaml`:
- `model.latent_dim`: Latent space dimensionality
- `model.hidden_dims`: Hidden layer sizes
- `trainer.max_epochs`: Training epochs
- `data.batch_size`: Batch size

#### Sampling

The trained VAE generates new, realistic attribute combinations:

```python
from src.vae.inference import VAESampler

sampler = VAESampler("models/vae_final.pth")
new_attributes = sampler.sample(n_samples=100)
```

### Stage 3: Caption Generation

**Notebook**: `4. Conditioned caption inference.ipynb`  
**Module**: `src/caption/`  
**Configs**: `config/caption_fine_tuning.yaml`, `config/caption_inference.yaml`

A fine-tuned LLM converts attribute lists into natural language descriptions.

#### Model

- Base model: LLaMA (configurable in `config/model/caption/`)
- Fine-tuning: LoRA adapters for efficiency
- Prompts: Templates in `config/prompt/`

#### Fine-tuning

```bash
python -m src.scripts.caption_fine_tuning
```

Key configurations:
- `lora.r`: LoRA rank
- `lora.alpha`: LoRA alpha scaling
- `model.base_model`: Base LLM name
- `data.max_length`: Maximum sequence length

#### Inference

```bash
python -m src.scripts.caption_inference
```

#### Prompt Templates

Available templates in `config/prompt/`:

| Template | Description |
|----------|-------------|
| `default.yaml` | Standard prompt format |
| `llama.yaml` | LLaMA-optimized prompts |
| `llama_short.yaml` | Shorter LLaMA prompts |
| `llama_zero_shot.yaml` | Zero-shot prompts |
| `mistral.yaml` | Mistral-optimized prompts |
| `qwen.yaml` | Qwen-optimized prompts |
| `gpt.yaml` | GPT-style prompts |

### Stage 4: Audio Generation

**Notebook**: `5. Audio generation analysis.ipynb`  
**Module**: `src/tta/`  
**Config**: `config/tta_inference.yaml`

MusicGen synthesizes audio from the generated captions.

#### Process

1. Load fine-tuned captions
2. Generate audio using MusicGen
3. Evaluate audio-text alignment

#### Inference

```bash
python -m src.scripts.tta_inference
```

Configuration options:
- `model.name`: MusicGen model variant
- `generation.duration`: Audio duration in seconds
- `generation.guidance_scale`: Classifier-free guidance scale

### Stage 5: Dataset Assembly

**Notebook**: `6. Create final datasets.ipynb`

Combines all components into the final dataset format.

#### Output Structure

```python
{
    "id": "unique_identifier",
    "caption": "Natural language description...",
    "aspect_list": "['tag1', 'tag2', 'tag3']",
    "genre_aspects": ["genre1", "genre2"],
    "mood_aspects": ["mood1"],
    "instrument_aspects": ["instrument1", "instrument2"],
    "tempo_aspects": ["tempo1"],
    "file_name": "path/to/audio.wav"  # For audio configurations
}
```

## Evaluation

### Caption Quality

**Module**: `src/caption/evaluation.py`

Metrics used:
- **BERTScore**: Semantic similarity
- **ROUGE**: N-gram overlap
- **MAUVE**: Distribution comparison

### Audio-Text Alignment

**Module**: `src/tta/evaluation.py`

Metrics used:
- **CLAP Score**: Contrastive audio-text alignment

### TCAV Analysis

**Notebook**: `7. TCAV for genre classification.ipynb`  
**Module**: `src/tcav/`

Tests whether concept probes recover musically meaningful patterns.

#### Process

1. Train genre classifier (`models/best-genre-classifier.ckpt`)
2. Extract concept activation vectors
3. Validate concept separability

## Configuration System

The project uses Hydra for configuration management.

### Running with Custom Configs

```bash
# Use specific preset
python -m src.scripts.caption_fine_tuning preset=caption/default

# Override parameters
python -m src.scripts.vae_training model.latent_dim=64 trainer.max_epochs=50

# Use different paths
python -m src.scripts.caption_inference paths=plgrid
```

### Config Groups

| Group | Purpose |
|-------|---------|
| `callbacks/` | Training callbacks |
| `data/` | Dataset configurations |
| `evaluation/` | Evaluation settings |
| `generation/` | Generation parameters |
| `logger/` | Logging (W&B) |
| `lora/` | LoRA hyperparameters |
| `model/` | Model architectures |
| `paths/` | Path configurations |
| `preset/` | Full experiment presets |
| `prompt/` | LLM prompt templates |
| `sweeps/` | Hyperparameter sweeps |
| `trainer/` | PyTorch Lightning trainer |

## Data Flow

```
┌─────────────────┐
│   MusicCaps     │
│   (5,521 clips) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tag Taxonomy   │
│  (200 attrs)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      VAE        │
│   (sampling)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuned LLM │
│  (captioning)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    MusicGen     │
│(audio synthesis)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ConceptCaps    │
│   (23k triplets)│
└─────────────────┘
```
