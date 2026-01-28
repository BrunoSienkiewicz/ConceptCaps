from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from src.constants import DEFAULT_GUIDANCE_SCALE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_SAMPLE_RATE
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm



from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def generate_audio_samples(
    model,
    dataloader: DataLoader,
    audio_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    df: pd.DataFrame,
    id_column: str = "id",
    filename_template: str = "{}.wav",
    temperature: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = True,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    loggers: list = None,
) -> None:
    os.makedirs(audio_dir, exist_ok=True)
    if sample_rate is None:
        # Derive sample rate from model config
        sample_rate = model.config.audio_encoder.sampling_rate

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Generating audio")
    ):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "guidance_scale": guidance_scale,
            "use_cache": True,  # Enable KV-cache for faster generation
        }
        audio_values = model.generate(**generation_kwargs)
        # Immediately move to CPU to free GPU memory
        audio_values_cpu = audio_values.cpu()
        del audio_values
        del input_ids
        del attention_mask
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if loggers:
            for logger in loggers:
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(
                        {"generated_samples": (batch_idx + 1) * batch_size},
                        step=batch_idx,
                    )

        for item_idx, audio in enumerate(audio_values_cpu):
            global_idx = batch_idx * batch_size + item_idx
            sample_id = df.iloc[global_idx][id_column]

            # Normalize/Clip audio to prevent artifacts
            audio_data = audio[0].float().cpu().numpy()
            audio_data = np.clip(audio_data, -1.0, 1.0)

            scipy.io.wavfile.write(
                audio_dir / filename_template.format(sample_id),
                sample_rate,
                audio_data,
            )


def generate_audio_samples_accelerate(
    model,
    dataloader: DataLoader,
    audio_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    df: pd.DataFrame,
    id_column: str = "id",
    filename_template: str = "{}.wav",
    temperature: float = 1.0,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = True,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    loggers: list = None,
) -> None:
    """Generate audio using Accelerate for multi-GPU distribution."""

    accelerator = Accelerator()

    # Debug: Print process and device info
    log.info(
        f"[Process {accelerator.process_index}] Local rank: {accelerator.local_process_index}, Device: {accelerator.device}"
    )

    os.makedirs(audio_dir, exist_ok=True)

    model, dataloader = accelerator.prepare(model, dataloader)

    if hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "guidance_scale": guidance_scale,
        "use_cache": True,
    }
    if sample_rate is None:
        # Derive sample rate from model config
        sample_rate = unwrapped_model.config.audio_encoder.sampling_rate

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc="Generating audio",
            disable=not accelerator.is_main_process,
        )
    ):
        input_ids, attention_mask = batch
        audio_values = unwrapped_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        audio_values = accelerator.gather(audio_values)
        # Immediately move to CPU to free GPU memory
        audio_values_cpu = audio_values.cpu()
        del audio_values
        del input_ids
        del attention_mask
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if loggers:
            for logger in loggers:
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(
                        {"generated_samples": (batch_idx + 1) * batch_size},
                        step=batch_idx,
                    )

        # Only main process saves files
        if accelerator.is_main_process:
            for item_idx, audio in enumerate(audio_values_cpu):
                # Calculate global index accounting for distributed batches
                global_idx = (
                    batch_idx * batch_size * accelerator.num_processes
                    + item_idx
                )
                audio_data = audio[0].float().cpu().numpy()

                # Peak normalization to prevent clipping
                max_val = np.abs(audio_data).max()
                if max_val > 0:
                    audio_data = audio_data / max(max_val, 1e-6)

                if global_idx < len(df):
                    sample_id = df.iloc[global_idx][id_column]
                    scipy.io.wavfile.write(
                        audio_dir / filename_template.format(sample_id),
                        sample_rate,
                        audio_data,
                    )

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
