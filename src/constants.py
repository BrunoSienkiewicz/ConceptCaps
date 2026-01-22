"""Project-wide constants and configuration values."""

import os
from pathlib import Path
from typing import Tuple

# ============================================================================
# Environment Variables
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.absolute()))

# Data directories
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

# Weights & Biases
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "music-gen-interpretability")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

# ============================================================================
# Audio Processing Constants
# ============================================================================

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION = 3.0
DEFAULT_N_MELS = 128
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_FFT = 2048

# ============================================================================
# GTZAN Dataset Constants
# ============================================================================

GTZAN_GENRES: Tuple[str, ...] = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)

GENRE_TO_IDX = {genre: idx for idx, genre in enumerate(GTZAN_GENRES)}
IDX_TO_GENRE = {idx: genre for idx, genre in enumerate(GTZAN_GENRES)}
NUM_GENRES = len(GTZAN_GENRES)

# ============================================================================
# Model Architecture Constants
# ============================================================================

DEFAULT_LATENT_DIM = 256
DEFAULT_HIDDEN_DIM = 128
DEFAULT_EMBEDDING_DIM = 512

# ============================================================================
# Training Constants
# ============================================================================

DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_GRADIENT_CLIP_VAL = 1.0

# ============================================================================
# TCAV Analysis Constants
# ============================================================================

MIN_CAV_ACCURACY = 0.60
DEFAULT_NUM_CAV_RUNS = 40
DEFAULT_NUM_CONCEPT_SAMPLES = 50
DEFAULT_NUM_RANDOM_SAMPLES = 100

# ============================================================================
# Text-to-Audio Generation Constants
# ============================================================================

DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 1503

# MusicGen Models
MUSICGEN_SMALL = "facebook/musicgen-small"
MUSICGEN_MEDIUM = "facebook/musicgen-medium"
MUSICGEN_LARGE = "facebook/musicgen-large"

# CLAP Models
CLAP_MODEL = "laion/clap-htsat-unfused"

# ============================================================================
# Evaluation Constants
# ============================================================================

# CLAP Score Benchmarks
CLAP_SCORE_EXCELLENT = 0.35  # Above this is excellent
CLAP_SCORE_GOOD = 0.25  # Above this is good
CLAP_SCORE_ACCEPTABLE = 0.15  # Above this is acceptable

# FAD Score (lower is better, these are approximate for CLAP-based FCD)
FAD_SCORE_EXCELLENT = 0.5
FAD_SCORE_GOOD = 1.0
FAD_SCORE_ACCEPTABLE = 2.0
