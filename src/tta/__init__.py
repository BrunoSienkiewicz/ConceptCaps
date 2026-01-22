"""Text-to-Audio generation module for MusicGen-based audio synthesis."""

from src.tta.audio import generate_audio_samples, generate_audio_samples_accelerate
from src.tta.config import TTAConfig
from src.tta.data import prepare_dataloader
from src.tta.evaluation import TTAEvaluator
from src.tta.model import prepare_model, prepare_tokenizer

__all__ = [
    "TTAConfig",
    "TTAEvaluator",
    "generate_audio_samples",
    "generate_audio_samples_accelerate",
    "prepare_dataloader",
    "prepare_model",
    "prepare_tokenizer",
]
