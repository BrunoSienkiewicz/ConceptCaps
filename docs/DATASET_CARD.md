---
annotations_creators:
- machine-generated
language_creators:
- machine-generated
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- google/MusicCaps
task_categories:
- text-generation
- text-to-audio
task_ids:
- natural-language-inference
- semantic-similarity-classification
pretty_name: ConceptCaps
tags:
- music
- audio
- captions
- text-to-audio
- music-generation
- interpretability
---

# Dataset Card for ConceptCaps

## Table of Contents
- [Dataset Card for ConceptCaps](#dataset-card-for-conceptcaps)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Overview](#overview)
    - [Key Features](#key-features)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Splits](#data-splits)
    - [Data Fields](#data-fields)
    - [Data Instances](#data-instances)
  - [Dataset Generation Pipeline](#dataset-generation-pipeline)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Curation Rationale](#curation-rationale)
    - [Annotations](#annotations)
      - [Used Models](#used-models)
      - [Annotation process](#annotation-process)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
  - [Usage Examples](#usage-examples)
    - [Load the default configuration (captions only):](#load-the-default-configuration-captions-only)
    - [Load with audio:](#load-with-audio)
    - [Load smaller subsets for quick experimentation:](#load-smaller-subsets-for-quick-experimentation)

## Dataset Description

- **Repository:** [GitHub Repository](https://github.com/BrunoSienkiewicz/ConceptCaps)
- **Paper:** [arXiv:2601.14157](https://arxiv.org/abs/2601.14157)

### Overview

ConceptCaps is a music captioning dataset derived from MusicCaps, specifically designed for concept-based interpretability research in text-to-audio (TTA) generation systems.
The dataset provides categorized musical concept annotations from distilled taxonomy (200 unique tags) alongside natural language captions, enabling fine-grained analysis of how TTA models represent and generate musical concepts.

The dataset is available in 2 versions: with and without audio.

### Key Features

- **21k music-caption-audio triplets** with explicit labels from a 200-attribute taxonomy
- **178 hours of audio content** paired with textual descriptions
- **Four concept categories**: genre, mood, instruments, tempo
- **Separated semantic modeling from text generation**: VAE learns attribute co-occurrence, LLM generates descriptions
- **Validated through multiple metrics**: CLAP alignment, BERTScore, MAUVE, and TCAV analysis

### Languages

The captions in ConceptCaps are in English (en).

## Dataset Structure

### Data Splits

| Configuration | Train | Validation | Test | Total |
|--------------|-------|------------|------|-------|
| default | 15003 | 3,215 | 3,215 | 21,433 |
| 25pct | 3,750 | 803 | 803 | 5,356 |
| 10pct | 1500 | 321 | 321 | 2,142 |

Splits follow a 70/15/15 ratio for train/validation/test.

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the sample |
| `caption` | string | Natural language description of the music |
| `aspect_list` | string | Stringified list of all musical concept tags |
| `genre_aspects` | list[string] | Genre-related tags (e.g., "jazz", "rock", "classical") |
| `mood_aspects` | list[string] | Mood/emotion tags (e.g., "mellow", "energetic", "sad") |
| `instrument_aspects` | list[string] | Instrument tags (e.g., "piano", "guitar", "drums") |
| `tempo_aspects` | list[string] | Tempo-related tags (e.g., "slow", "fast", "moderate") |
| `file_name` | Audio | (Audio versions only) Audio file data |

### Data Instances

A typical data instance looks like:

```json
{
  "id": "b5fb15e8252105205ac5fb8053745993",
  "caption": "This slow pop-rock track features a melancholic guitar-driven arrangement at a relaxed pace, accompanied only by minimalist instrumental textures without any percussive elements or vocal components. [...]",
  "aspect_list": "['guitar', 'no percussion', 'no voices', 'pop', 'slow rock', 'slow tempo']",
  "genre_aspects": ["pop", "slow rock"],
  "mood_aspects": [],
  "instrument_aspects": ["guitar", "no percussion", "no voices"],
  "tempo_aspects": ["slow tempo"]
}
```

For audio versions, an additional `file_name` field contains the audio data.

## Dataset Generation Pipeline

![](./assets/pipeline.pdf)

### Source Data

#### Initial Data Collection and Normalization

ConceptCaps is derived from [MusicCaps](https://huggingface.co/datasets/google/MusicCaps), a dataset of 5,521 music clips with expert-written captions from YouTube. 
The original `aspect_list` annotations were systematically filtered and categorized into four concept categories to create curated taxonomy.
Using this taxonomy we curated original MusicCaps dataset to create pairs of `aspect_list` and `caption` used for downstream training tasks.

#### Curation Rationale

ConceptCaps was created to enable interpretability research in text-to-audio generation.
Existing music captioning datasets contain noisy or sparse data, making it difficult to perform concept interpretability research.
By distillation and categorization of musical aspects (genre, mood, instruments, tempo), ConceptCaps provides strong foundation for various interpretability methods in music.

### Annotations

#### Used Models

#### Annotation process

1. **Concept Extraction**: Tags from MusicCaps `aspect_list` were mapped to four categories (genre, mood, instrument, tempo) using a curated taxonomy
2. **Tag Generation**: Tag combinations were generated using custom VAE trained on the curated dataset tag combinations
3. **Caption Extrapolation**: A fine-tuned LLM generated natural language captions conditioned on the obtained annotation combinations
4. **Audio Inference**: Audio samples were inferenced using extrapolated captions 

## Additional Information

### Dataset Curators

This dataset was created by Bruno Sienkiewicz, Łukasz Neumann, and Mateusz Modrzejewski as part of research on interpretability in text-to-audio generation systems.

### Licensing Information

This dataset is released under the [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/).

### Citation Information

If you use ConceptCaps in your research, please cite:

```bibtex
@article{sienkiewicz2026conceptcaps,
  title={ConceptCaps -- a Distilled Concept Dataset for Interpretability in Music Models},
  author={Sienkiewicz, Bruno and Neumann, Łukasz and Modrzejewski, Mateusz},
  journal={arXiv preprint arXiv:2601.14157},
  year={2026}
}
```

## Usage Examples

### Load the default configuration (captions only):

```python
from datasets import load_dataset

dataset = load_dataset("bsienkiewicz/ConceptCaps", "default")
print(dataset["train"][0])
```

### Load with audio:

```python
dataset = load_dataset("bsienkiewicz/ConceptCaps", "audio")
# Access audio
audio_data = dataset["train"][0]["file_name"]
```

### Load smaller subsets for quick experimentation:

```python
# 10% subset without audio
dataset_small = load_dataset("bsienkiewicz/ConceptCaps", "10pct")

# 25% subset with audio
dataset_medium = load_dataset("bsienkiewicz/ConceptCaps", "25pct-audio")
```