# VAE Metrics Saving and Configuration Comparison

This document explains how to save metrics from VAE training runs and compare multiple configurations.

## Metrics Saving

### Automatic Metrics Saving

Metrics are automatically saved at the end of training in the following locations:

```
outputs/vae_training/{run_id}/
├── metrics.json          # Raw metrics from all phases
├── summary.json          # Metrics + configuration used
└── ...

models/metrics/
├── {model_name}_metrics.json    # Copy of metrics
└── {model_name}_summary.json    # Copy of summary
```

### Metrics Structure

The saved JSON files contain metrics organized by training phase:

```json
{
  "train": {
    "train_loss": 1234.5,
    "train_bce": 890.3,
    "train_kld": 344.2,
    ...
  },
  "val": {
    "val_loss": 1200.1,
    "val_bce": 870.5,
    "val_kld": 329.6,
    "val_hamming_loss": 0.15,
    "val_jaccard_index": 0.68,
    ...
  },
  "test": {
    "test_loss": 1195.8,
    "test_bce": 865.2,
    "test_kld": 330.6,
    "test_diversity_pct_unique_combinations": 87.5,
    "test_diversity_entropy": 8.3,
    "test_cooccurrence_cosine_similarity": 0.82,
    ...
  }
}
```

### Summary File Structure

The summary file includes both configuration and metrics:

```json
{
  "config": {
    "model_name": "vae_small",
    "model": {
      "latent_dim": 32,
      "hidden_dim": 128,
      "dropout_p": 0.3
    },
    "trainer": {
      "max_epochs": 250,
      "learning_rate": 5e-4
    },
    ...
  },
  "metrics": {
    "train": {...},
    "val": {...},
    "test": {...}
  }
}
```

## Configuration Comparison

### Running Preset Configurations

Run a set of predefined configurations and compare results:

```bash
# Run standard VAE variants (small, medium, large)
./scripts/run_vae_comparison.sh --preset standard

# Run Beta-VAE variants
./scripts/run_vae_comparison.sh --preset beta

# Run all configurations
./scripts/run_vae_comparison.sh --preset all
```

### Running Custom Configurations

```bash
# Single custom configuration
./scripts/run_vae_comparison.sh --config my_vae model=vae/small trainer=vae/default

# Multiple custom configurations
./scripts/run_vae_comparison.sh \
  --config vae_custom1 model=vae/small epochs=100 \
  --config vae_custom2 model=vae/medium epochs=200
```

### Collecting Existing Results

If training runs already exist, collect and compare without running new training:

```bash
# Compare all existing results
./scripts/run_vae_comparison.sh --collect-only

# Compare specific metrics directory
./scripts/run_vae_comparison.sh --collect-only --metrics-dir ./models/metrics
```

### Python API

Use the Python API directly for more control:

```python
from src.scripts.vae.run_vae_comparison import VAEExperimentRunner
from pathlib import Path

runner = VAEExperimentRunner(project_root=Path("."))

# Run single configuration
runner.run_configuration(
    config_name="my_experiment",
    model_variant="small",
    trainer_variant="default",
    epochs=300,
    batch_size=64,
    lr=1e-3
)

# Collect results
results = runner.collect_results(metrics_dir=Path("models/metrics"))

# Create comparison
df = runner.compare_configurations(results)
runner.save_comparison(df)

# Generate report
report = runner.generate_summary_report(df)
print(report)
```

## Comparison Output

### Comparison CSV

The comparison is saved as `outputs/vae_experiments/comparison.csv` with columns:

- `configuration`: Configuration name
- `test_loss`: Final test loss
- `test_bce`: Final BCE loss
- `test_kld`: Final KL divergence
- `hamming_loss`: Reconstruction error
- `jaccard_index`: Multi-label accuracy
- `pct_active_units`: % of active latent dimensions
- `silhouette_score`: Cluster separability
- `pct_unique_combinations`: Diversity of tag combinations
- `entropy`: Entropy of combination distribution
- `gini_coefficient`: Inequality of combinations
- `cosine_similarity`: Co-occurrence pattern similarity
- `kl_divergence`: Co-occurrence pattern divergence

### Comparison Report

A human-readable text report is saved as `outputs/vae_experiments/comparison_report.txt` showing:

1. **Loss Metrics**: Best configuration for each loss component
2. **Reconstruction Metrics**: Best Hamming loss and Jaccard index
3. **Latent Space Metrics**: Best active units and silhouette score
4. **Diversity Metrics**: Best diversity and entropy
5. **Co-occurrence Metrics**: Best pattern similarity

Example report excerpt:
```
====================================================================================================
VAE CONFIGURATION COMPARISON REPORT
====================================================================================================

Total configurations evaluated: 6

----------------------------------------------------------------------------------------------------
Loss Metrics (lower is better)
----------------------------------------------------------------------------------------------------
test_loss                 : Best=vae_medium                 (1195.3200)
test_bce                  : Best=vae_medium                 (  865.1200)
test_kld                  : Best=vae_medium                 (  330.2000)

----------------------------------------------------------------------------------------------------
Reconstruction Metrics
----------------------------------------------------------------------------------------------------
hamming_loss              : Best=vae_large                  (    0.1200)
jaccard_index             : Best=beta_vae_balanced          (    0.7850)

----------------------------------------------------------------------------------------------------
Diversity Metrics (higher is better for entropy, lower for Gini)
----------------------------------------------------------------------------------------------------
pct_unique_combinations   : Best=beta_vae_high              (   89.3200)
entropy                   : Best=beta_vae_high              (    8.5100)
gini_coefficient          : Best=beta_vae_balanced          (    0.1850)
```

## Workflow Example

### Step 1: Run Experiments

```bash
# Run all standard and beta-vae configurations
./scripts/run_vae_comparison.sh --preset all
```

This runs 6 configurations sequentially and saves metrics.

### Step 2: Review Comparison

After training completes:

```bash
# Compare results
cat outputs/vae_experiments/comparison_report.txt

# View detailed CSV
less outputs/vae_experiments/comparison.csv
```

### Step 3: Select Best Configuration

Based on the comparison, select the best configuration for your use case:

- **Best reconstruction**: Hamming loss and Jaccard index
- **Best diversity**: Entropy and unique combinations
- **Best latent space**: Active units and silhouette score
- **Best pattern matching**: Cosine similarity and KL divergence

### Step 4: Run Final Model

```bash
# Train the best configuration for longer
./scripts/run_vae_training.sh --model large --epochs 500 --batch-size 32
```

## Directory Structure

After running comparisons:

```
models/
├── metrics/
│   ├── vae_small_metrics.json
│   ├── vae_small_summary.json
│   ├── vae_medium_metrics.json
│   ├── vae_medium_summary.json
│   ├── vae_large_metrics.json
│   ├── vae_large_summary.json
│   ├── beta_vae_default_metrics.json
│   ├── beta_vae_default_summary.json
│   ├── beta_vae_balanced_metrics.json
│   ├── beta_vae_balanced_summary.json
│   ├── beta_vae_high_metrics.json
│   └── beta_vae_high_summary.json
├── vae_small.pth
├── vae_medium.pth
├── vae_large.pth
└── ...

outputs/
└── vae_experiments/
    ├── comparison.csv
    └── comparison_report.txt
```

## Tips for Configuration Comparison

1. **Run with consistent settings**: Use the same number of epochs and batch size for fair comparison
2. **Monitor multiple metrics**: Don't optimize for just one metric
3. **Check diversity**: Models with low diversity indicate collapse
4. **Validate co-occurrence**: Ensure tag patterns match the original data
5. **Consider latent space**: Active units should be reasonable (not too many, not too few)

## Common Comparisons

### VAE vs Beta-VAE

```bash
./scripts/run_vae_comparison.sh \
  --config "standard_vae" model=vae/medium trainer=vae/default \
  --config "beta_vae_4" model=vae/beta_balanced beta=4.0 \
  --config "beta_vae_10" model=vae/beta_balanced beta=10.0
```

**Interpretation**:
- Beta-VAE should have higher entropy and more unique combinations
- Standard VAE should have lower KL divergence
- Beta-VAE should have better latent factor disentanglement

### Architecture Comparison

```bash
./scripts/run_vae_comparison.sh --preset standard
```

**Interpretation**:
- Larger models typically have lower reconstruction loss
- Smaller models are faster and use less memory
- Medium models often provide best balance

### Hyperparameter Tuning

```bash
./scripts/run_vae_comparison.sh \
  --config "lr_5e4" model=vae/medium trainer=vae/default lr=5e-4 \
  --config "lr_1e3" model=vae/medium trainer=vae/default lr=1e-3 \
  --config "lr_1e4" model=vae/medium trainer=vae/default lr=1e-4
```

**Interpretation**:
- Different learning rates affect convergence speed
- Too high LR may cause instability
- Too low LR may result in underfitting
