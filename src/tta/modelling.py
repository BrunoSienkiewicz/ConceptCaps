
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer
from transformers import MusicgenForConditionalGeneration
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

def prepare_tokenizer(model_cfg: DictConfig) -> AutoTokenizer:
    tokenizer_kwargs: dict[str, object] = {}
    tokenizer_cfg = model_cfg.get("tokenizer", {})
    if tokenizer_cfg.get("use_fast") is not None:
        tokenizer_kwargs["use_fast"] = tokenizer_cfg["use_fast"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, **tokenizer_kwargs)
    if tokenizer_cfg.get("padding_side"):
        tokenizer.padding_side = tokenizer_cfg["padding_side"]
    if tokenizer_cfg.get("pad_token_as_eos", True) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_model(model_cfg: DictConfig) -> MusicgenForConditionalGeneration:
    # Load model with optimizations
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_cfg.name,
        device_map=model_cfg.device_map,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=torch.bfloat16 if model_cfg.get("use_bf16", True) else torch.float32,
    )
    
    # Enable gradient checkpointing to save memory
    if model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")
    
    # Compile model for faster generation (PyTorch 2.0+)
    if model_cfg.get("compile", False) and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
    
    return model
