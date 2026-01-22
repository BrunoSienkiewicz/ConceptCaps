"""Model preparation utilities for text-to-audio generation."""

from typing import Optional

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, MusicgenForConditionalGeneration

from src.constants import DEFAULT_GUIDANCE_SCALE, MUSICGEN_LARGE
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def prepare_tokenizer(model_cfg: DictConfig) -> AutoTokenizer:
    """Prepare and configure tokenizer for MusicGen model.

    Args:
        model_cfg: Model configuration with tokenizer settings.

    Returns:
        Configured AutoTokenizer instance.
    """
    tokenizer_kwargs: dict[str, object] = {}
    tokenizer_cfg = model_cfg.get("tokenizer", {})
    if tokenizer_cfg.get("use_fast") is not None:
        tokenizer_kwargs["use_fast"] = tokenizer_cfg["use_fast"]

    log.info(f"Loading tokenizer from {model_cfg.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, **tokenizer_kwargs
    )
    if tokenizer_cfg.get("padding_side"):
        tokenizer.padding_side = tokenizer_cfg["padding_side"]
    if (
        tokenizer_cfg.get("pad_token_as_eos", True)
        and tokenizer.pad_token is None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Tokenizer configured successfully")
    return tokenizer


def prepare_model(model_cfg: DictConfig) -> MusicgenForConditionalGeneration:
    """Load and configure MusicGen model for audio generation.

    Args:
        model_cfg: Model configuration including name, device_map, and optimization settings.

    Returns:
        Configured MusicgenForConditionalGeneration model.
    """
    log.info(f"Loading model: {model_cfg.name}")

    # Load model with optimizations
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_cfg.name,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=torch.bfloat16
        if model_cfg.get("use_bf16", True)
        else torch.float32,
    )

    log.info(f"Model loaded on device_map: {model_cfg.device_map}")

    # Enable gradient checkpointing to save memory
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")

    # Compile model for faster generation (PyTorch 2.0+)
    if model_cfg.get("compile", False) and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        log.info("Model compilation complete")

    log.info("Model preparation complete")
    return model
