# ConceptCaps Documentation

Welcome to the ConceptCaps documentation. ConceptCaps is a distilled concept dataset for interpretability research in music models.

## Quick Links

- **Paper**: [arXiv:2601.14157](https://arxiv.org/abs/2601.14157)
- **Dataset**: [Hugging Face](https://huggingface.co/datasets/bsienkiewicz/ConceptCaps)
- **Repository**: [GitHub](https://github.com/BrunoSienkiewicz/ConceptCaps)

## Documentation Contents

### Getting Started

- [README](https://github.com/BrunoSienkiewicz/ConceptCaps/blob/main/README.md) - Project overview and installation

## Overview

ConceptCaps is a concept-based music captioning dataset designed for interpretability research in text-to-audio (TTA) generation systems. The dataset provides:

- **21,000+ music-caption-audio triplets** with explicit labels
- **200-attribute taxonomy** covering genre, mood, instruments, and tempo
- **Structured annotations** for concept-based analysis

## Pipeline Summary

```
MusicCaps → Taxonomy Extraction → VAE Sampling → LLM Captioning → MusicGen Audio
```

1. **Taxonomy Extraction**: Map MusicCaps tags to structured categories
2. **VAE Sampling**: Learn and sample plausible attribute combinations
3. **LLM Captioning**: Generate natural language descriptions from attributes
4. **Audio Synthesis**: Create audio using MusicGen

## Key Features

### Concept Separation

Unlike existing datasets, ConceptCaps provides clean separation between:
- Genre concepts (jazz, rock, classical, etc.)
- Mood concepts (happy, sad, energetic, etc.)
- Instrument concepts (piano, guitar, drums, etc.)
- Tempo concepts (slow, fast, moderate, etc.)

### TCAV Compatibility

The dataset is designed for Testing with Concept Activation Vectors (TCAV), enabling:
- Concept probe training
- Model interpretability analysis
- Controlled generation experiments

### Quality Validation

All data is validated through:
- **CLAP**: Audio-text alignment scores
- **BERTScore**: Caption semantic quality
- **MAUVE**: Distribution comparison
- **TCAV**: Concept separability verification

## Citation

```bibtex
@article{sienkiewicz2026conceptcaps,
  title={ConceptCaps: a Distilled Concept Dataset for Interpretability in Music Models},
  author={Sienkiewicz, Bruno and Neumann, Łukasz and Modrzejewski, Mateusz},
  journal={arXiv preprint arXiv:2601.14157},
  year={2026}
}
```

## License

This project is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
