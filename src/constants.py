import os
from pathlib import Path
from typing import Tuple

# ===========================================================================
# Final dataset paths
# ===========================================================================

METADATA_CSV_PATH = Path(os.getenv("METADATA_CSV_PATH", "../data/generated_audio_dataset/metadata.csv"))
AUDIO_DATA_PATH = Path(os.getenv("AUDIO_DATA_PATH", "../data/generated_audio_dataset"))

TCAV_RESULTS_PATH = Path(os.getenv("TCAV_RESULTS_PATH", "../data/tcav_genre_classification_results.json"))

# ============================================================================
# Environment Variables
# ============================================================================

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).parent.parent.absolute()))

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

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

GTZAN_PATH = Path(os.getenv("GTZAN_PATH", "../data/GTZAN/Data/genres_original"))
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

GENRE_CLASSIFIER_MODEL_CHECKPOINT_PATH = Path(os.getenv("GENRE_CLASSIFIER_MODEL_CHECKPOINT_PATH", "../models/best-genre-classifier.ckpt"))

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

MUSICGEN_SMALL = "facebook/musicgen-small"
MUSICGEN_MEDIUM = "facebook/musicgen-medium"
MUSICGEN_LARGE = "facebook/musicgen-large"

CLAP_MODEL = "laion/clap-htsat-unfused"
