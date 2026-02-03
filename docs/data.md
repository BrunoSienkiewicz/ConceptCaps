# Data

Datasets and intermediate data files.

## Directory Structure

```
data/
├── concepts_to_tags.json              # Taxonomy mapping
├── musiccaps_tag_frequencies.csv      # Tag frequencies in MusicCaps
├── tcav_genre_classification_results.json  # TCAV analysis results
│
├── evaluation_results/                # Metric outputs
│   ├── base.csv                       # Base LLM results
│   ├── ft.csv                         # Fine-tuned LLM results
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
├── musiccaps-tags-to-caption/         # DIstilled MusicCaps dataset
│   └── all.csv
│
├── random-tags-dataset/               # Random tag combinations
└── vae-tags-dataset/                  # VAE-sampled combinations
```
