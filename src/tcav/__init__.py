"""TCAV (Testing with Concept Activation Vectors) module for interpretability analysis.

Provides tools to analyze how neural networks use high-level concepts in their predictions
by training linear classifiers (CAVs) on activations and computing directional derivatives.
"""

from src.tcav.model import MusicGenreClassifier
from src.tcav.tcav import TCAV

__all__ = [
    "MusicGenreClassifier",
    "TCAV",
]