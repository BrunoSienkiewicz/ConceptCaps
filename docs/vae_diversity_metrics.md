# VAE Diversity and Co-occurrence Metrics

This document describes the diversity and co-occurrence metrics computed during VAE model evaluation, particularly in the test phase.

## Overview

The VAE training system evaluates model quality through multiple lenses:

1. **Reconstruction Quality**: How well the model reconstructs input data
2. **Latent Space Quality**: How well the latent space captures meaningful factors
3. **Diversity**: Whether the model generates diverse tag combinations or collapses to a few
4. **Consistency**: Whether the model respects tag co-occurrence patterns in the original data

## Diversity Metrics

### Tag Combination Diversity

Measures the diversity of generated tag combinations to detect model collapse.

#### Metrics:

- **Unique Combinations**: Number of distinct tag combinations in the batch
- **Percentage Unique Combinations**: Fraction of unique combinations in the batch (0-100%)
- **Entropy**: Shannon entropy of the combination distribution
  - Higher entropy indicates more diverse combinations
  - $H = -\sum_{i} p_i \log(p_i)$
  - Range: [0, log(N)] where N is max possible combinations
- **Gini Coefficient**: Inequality measure of combination distribution
  - 0 = perfectly uniform distribution (all combinations appear equally)
  - 1 = perfect concentration (all samples have one combination)
  - Formula: $G = \frac{2\sum_{i=1}^{n} i \cdot c_i}{n \sum c_i} - \frac{n+1}{n}$

### Interpretation:

- **Good Model**: High unique combinations %, high entropy, low Gini coefficient
- **Model Collapse**: Low unique combinations %, low entropy, high Gini coefficient
- **Warning Signs**:
  - `pct_unique_combinations < 50%` suggests potential collapse
  - `gini_coefficient > 0.5` suggests dominant combinations

## Co-occurrence Analysis

### Tag Co-occurrence Matrix

A matrix where entry [i, j] represents how many times tags i and j appear together in the batch.

**Shape**: (num_tags, num_tags)

Example:
```
Tags:      A  B  C
A  [10  5  3]
B  [ 5  8  2]
C  [ 3  2  6]
```

This means:
- Tag A appears 10 times
- Tags A and B appear together 5 times
- Tags B and C appear together 2 times

### Co-occurrence Comparison Metrics

Compare predicted co-occurrence patterns with the original dataset:

#### Metrics:

1. **Cosine Similarity** (higher is better)
   - Measure: $\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{|\mathbf{u}||\mathbf{v}|}$
   - Range: [-1, 1], typically [0, 1] for non-negative matrices
   - Interpretation: How similar the overall co-occurrence patterns are
   - Good range: > 0.7

2. **KL Divergence** (lower is better)
   - Measure: $D_{KL}(P||Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$
   - Range: [0, ∞)
   - Interpretation: How different the predicted distribution is from the original
   - Good range: < 0.5

3. **Hellinger Distance** (lower is better)
   - Measure: $H(P,Q) = \frac{1}{\sqrt{2}} \sqrt{\sum (\sqrt{p_i} - \sqrt{q_i})^2}$
   - Range: [0, 1]
   - Interpretation: Statistical distance between distributions
   - Good range: < 0.2

4. **Pearson Correlation** (higher is better)
   - Measure: Standard Pearson correlation coefficient
   - Range: [-1, 1]
   - Interpretation: Linear relationship between co-occurrence patterns
   - Good range: > 0.6

## Example Interpretation

### Good Model:
```
diversity_unique_combinations: 5000
diversity_pct_unique_combinations: 85.5
diversity_entropy: 8.2
diversity_gini_coefficient: 0.15
cooccurrence_cosine_similarity: 0.82
cooccurrence_kl_divergence: 0.23
cooccurrence_hellinger_distance: 0.12
cooccurrence_pearson_correlation: 0.78
```

**Interpretation**:
- Model generates diverse combinations (85.5% unique)
- Entropy and Gini indicate good diversity
- Co-occurrence patterns closely match the original data (high cosine, low KL)
- Good correlation with original patterns

### Poor Model (Collapsed):
```
diversity_unique_combinations: 50
diversity_pct_unique_combinations: 8.5
diversity_entropy: 1.2
diversity_gini_coefficient: 0.92
cooccurrence_cosine_similarity: 0.35
cooccurrence_kl_divergence: 2.45
cooccurrence_hellinger_distance: 0.68
cooccurrence_pearson_correlation: 0.12
```

**Interpretation**:
- Model generates very few unique combinations (8.5% unique)
- Very low entropy and high Gini indicate severe collapse
- Co-occurrence patterns don't match original data
- Weak correlation with original patterns

## Evaluation Strategy

During training, monitor these metrics to:

1. **Detect Collapse Early**: Watch entropy and Gini coefficient in validation
2. **Verify Reconstruction Quality**: Use Hamming loss and Jaccard index
3. **Check Latent Quality**: Monitor active units and silhouette score
4. **Final Evaluation**: Use diversity and co-occurrence metrics in test phase

## Configuration

Set these in test evaluation or validation callback:

```yaml
# Default threshold for binarization
threshold: 0.5

# If threshold too low: may count noise as valid tags
# If threshold too high: may miss valid tags
```

## References

- Entropy: Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- Gini Coefficient: Gini, C. (1912). "Variabilità e mutabilità"
- Silhouette Score: Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"
- Hellinger Distance: Hellinger, E. (1909). "Neue Begründung der Theorie quadratischer Formen von unendlichvielen Veränderlichen"
