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
- 1K<n<10K
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
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)
  - [Usage Examples](#usage-examples)
    - [Load the default configuration (captions only):](#load-the-default-configuration-captions-only)
    - [Load with audio:](#load-with-audio)
    - [Load smaller subsets for quick experimentation:](#load-smaller-subsets-for-quick-experimentation)

## Dataset Description

- **Repository:** [GitHub Repository]()
- **Paper:** [arXiv:2601.14157](https://arxiv.org/abs/2601.14157)

### Dataset Summary

ConceptCaps is a concept-based music captioning dataset derived from MusicCaps, designed for interpretability research in text-to-audio (TTA) generation systems. The dataset provides structured musical concept annotations alongside natural language captions, enabling fine-grained analysis of how TTA models represent and generate musical concepts.

The dataset is available in multiple configurations:
- **default**: Full dataset (5,358 samples) with captions only
- **25pct**: 25% subset (1,339 samples) with captions only
- **10pct**: 10% subset (535 samples) with captions only
- **audio**: Full dataset with audio files
- **25pct-audio**: 25% subset with audio files
- **10pct-audio**: 10% subset with audio files

### Supported Tasks and Leaderboards

ConceptCaps supports the following tasks:

- **Music Captioning**: Generate natural language descriptions from musical concept tags
- **Text-to-Audio Generation**: Generate audio from concept-conditioned captions
- **Concept-based Interpretability**: Analyze how TTA models encode musical concepts (genre, mood, instruments, tempo)

### Languages

The captions in ConceptCaps are in English (en).

## Dataset Structure

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

### Data Splits

| Configuration | Train | Validation | Test | Total |
|--------------|-------|------------|------|-------|
| default | 3,750 | 804 | 804 | 5,358 |
| 25pct | 937 | 201 | 201 | 1,339 |
| 10pct | 375 | 80 | 80 | 535 |
| audio | 3,750 | 804 | 804 | 5,358 |
| 25pct-audio | 937 | 201 | 201 | 1,339 |
| 10pct-audio | 375 | 80 | 80 | 535 |

Splits follow a 70/15/15 ratio for train/validation/test.

## Dataset Creation

### Curation Rationale

ConceptCaps was created to enable interpretability research in text-to-audio generation. Existing music captioning datasets lack structured concept annotations needed to systematically study how TTA models represent musical concepts. By providing explicit categorization of musical aspects (genre, mood, instruments, tempo), ConceptCaps facilitates:

1. Concept-conditioned caption generation
2. TCAV (Testing with Concept Activation Vectors) analysis
3. Controlled evaluation of TTA model behavior

### Source Data

#### Initial Data Collection and Normalization

ConceptCaps is derived from [MusicCaps](https://huggingface.co/datasets/google/MusicCaps), a dataset of 5,521 music clips with expert-written captions from YouTube. The original `aspect_list` annotations were systematically categorized into four concept categories using a curated taxonomy.

#### Who are the source language producers?

The original MusicCaps captions were written by professional musicians. The concept categorization and caption generation in ConceptCaps were produced using a fine-tuned language model conditioned on the structured concept tags.

### Annotations

#### Annotation process

1. **Concept Extraction**: Tags from MusicCaps `aspect_list` were mapped to four categories (genre, mood, instrument, tempo) using a manually curated taxonomy
2. **Caption Generation**: A fine-tuned LLM generated natural language captions conditioned on the categorized concept tags

#### Who are the annotators?

The concept taxonomy was created by the dataset curators. Caption generation was performed by a fine-tuned language model.

## Considerations for Using the Data

### Social Impact of Dataset

ConceptCaps is intended for research in music AI interpretability. The dataset could help:
- Improve transparency in AI music generation systems
- Enable better control over generated music content
- Support research into AI fairness and bias in music representation

### Discussion of Biases

The dataset inherits biases from MusicCaps, which:
- May underrepresent certain musical genres or cultures
- Contains primarily Western music
- Has annotations from a limited pool of musicians

### Other Known Limitations

- Audio is generated, not original recordings
- Concept taxonomy may not capture all musical nuances
- Caption quality depends on the fine-tuned model performance

## Additional Information

### Dataset Curators

This dataset was created by Bruno Sienkiewicz as part of research on interpretability in text-to-audio generation systems.

### Licensing Information

This dataset is released under the [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/).

### Citation Information

If you use ConceptCaps in your research, please cite:

```bibtex
```

### Contributions

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