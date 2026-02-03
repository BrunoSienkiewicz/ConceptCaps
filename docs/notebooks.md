# Notebooks

Jupyter notebooks demonstrating each pipeline stage.

## Pipeline Overview

The notebooks follow the dataset creation pipeline sequentially:

```
1. Taxonomy → 2. VAE → 3. Comparison → 4. Captions → 5. Audio → 6. Final → 7. TCAV
```

## Notebook Descriptions

### 1. Taxonomy and Dataset Distillation

**File**: `1. Taxonomy and dataset distillation.ipynb`

Creates the 200-tag taxonomy from MusicCaps annotations:

- Extract and clean tags from MusicCaps
- Map tags to concept categories (genre, mood, instrument, tempo)
- Analyze tag co-occurrence patterns
- Export `concepts_to_tags.json`

### 2. VAE Aspect Modeling

**File**: `2. VAE aspect modeling.ipynb`

Trains the Beta-VAE on tag distributions:

- Prepare multi-hot encoded tag vectors
- Visualize latent space
- Sample new attribute combinations

### 3. MusicCaps and VAE Generated Dataset Comparison

**File**: `3. MusicCaps and VAE generated dataset comparison.ipynb`

Validates VAE sampling quality:

- Compare tag distributions (original vs. sampled)
- Analyze co-occurrence preservation
- Statistical tests for distribution similarity

### 4. Conditioned Caption Inference

**File**: `4. Conditioned caption inference.ipynb`

Evaluates captions from the fine-tuned LLM against baselines:

- Analyze generated captions from VAE-sampled attributes
- Evaluate caption quality (perplexity, diversity)

### 5. Audio Generation Analysis

**File**: `5. Audio generation analysis.ipynb`

Analyzes MusicGen audio synthesis:

- Compute CLAP and FAD scores
- Quality assessment

### 6. Create Final Datasets

**File**: `6. Create final datasets.ipynb`

Assembles the final ConceptCaps dataset:

- Combine captions, audio paths, and metadata
- Create train/validation/test splits
- Export to HuggingFace format

### 7. TCAV for Genre Classification

**File**: `7. TCAV for genre classification.ipynb`

Interpretability analysis with TCAV:

- Train genre classifier on GTZAN
- Extract bottleneck activations
- Train CAVs for each concept
- Compute TCAV scores
- Visualize concept importance

