# Data

Datasets and intermediate data files.

## Directory Structure

```
data/
├── concepts_to_tags.json              # Taxonomy mapping
├── musiccaps_tag_frequencies.csv      # Tag statistics
├── tcav_genre_classification_results.json  # TCAV analysis results
│
├── evaluation_results/                # Metric outputs
│   ├── base.csv                       # Baseline model results
│   ├── ft.csv                         # Fine-tuned model results
│   └── zero_shot.csv                  # Zero-shot results
│
├── generated_captions/                # LLM outputs
│   ├── final_train.csv
│   ├── final_validation.csv
│   └── final_test.csv
│
├── mtg_jamendo/                       # MTG-Jamendo dataset
│   └── autotagging_top50tags_processed_cleaned.csv
│
├── musiccaps-tags-to-caption/         # MusicCaps processed
│   └── all.csv
│
├── random-tags-dataset/               # Random tag combinations
└── vae-tags-dataset/                  # VAE-sampled combinations
```

## Key Files

### `concepts_to_tags.json`

Maps the 200-tag taxonomy to four concept categories:

```json
{
  "genre": ["jazz", "rock", "classical", ...],
  "mood": ["happy", "sad", "energetic", ...],
  "instrument": ["piano", "guitar", "drums", ...],
  "tempo": ["slow", "moderate", "fast", ...]
}
```

### `musiccaps_tag_frequencies.csv`

Tag occurrence statistics from MusicCaps for sampling weights.

| Column | Description |
|--------|-------------|
| `tag` | Tag name |
| `count` | Occurrence count |
| `frequency` | Relative frequency |

### `tcav_genre_classification_results.json`

TCAV scores for each concept-genre pair. Structure:

```json
{
  "concept_name": {
    "target_genre": {
      "tcav_score": 0.85,
      "cav_accuracy": 0.92,
      "p_value": 0.001
    }
  }
}
```

## Generated Outputs

### Caption Files

CSV files with columns:

| Column | Description |
|--------|-------------|
| `id` | Sample identifier |
| `aspects` | Tag list (JSON array) |
| `caption` | Generated caption |
| `split` | train/validation/test |

### Evaluation Results

Metrics computed for each model variant:

- **CLAP**: Audio-text alignment score
- **BERTScore**: Semantic similarity to references
- **MAUVE**: Distribution quality metric
