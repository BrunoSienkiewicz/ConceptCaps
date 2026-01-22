"""Music Generation Interpretability Research Package.

This package provides tools for analyzing and interpreting music generation
models, including VAE-based representation learning, TCAV analysis, and text-to-
audio generation.
"""

__version__ = "0.1.0"
__author__ = "Bruno Sienkiewicz"

from src.constants import DATA_DIR, GTZAN_GENRES, MODELS_DIR, OUTPUTS_DIR

__all__ = [
    "DATA_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "GTZAN_GENRES",
]
