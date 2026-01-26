from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
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
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    guidance_scale: float = None,
    sample_rate: int = 32000,
    loggers: list = None,
) -> None:
    os.makedirs(audio_dir, exist_ok=True)

    for batch_idx, batch in enumerate(
        tqdm(dataloader, desc="Generating audio")
    ):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.inference_mode():
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "do_sample": do_sample,
                "use_cache": True,  # Enable KV-cache for faster generation
            }
            if guidance_scale is not None:
                generation_kwargs["guidance_scale"] = guidance_scale

            audio_values = model.generate(**generation_kwargs)

        if loggers:
            for logger in loggers:
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(
                        {"generated_samples": (batch_idx + 1) * batch_size},
                        step=batch_idx,
                    )

        for item_idx, audio in enumerate(audio_values):
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

        # Clear CUDA cache after each batch to prevent memory fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    guidance_scale: float = None,
    sample_rate: int = 32000,
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
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "audio_channels": 1,  # Mono can be cleaner
        "extend_stride": 18,  # Overlap for smoother continuity (in seconds)
        "use_cache": True,
    }

    if guidance_scale is not None:
        generation_kwargs["guidance_scale"] = guidance_scale

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc="Generating audio",
            disable=not accelerator.is_main_process,
        )
    ):
        input_ids, attention_mask = batch

        with torch.inference_mode():
            audio_values = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        # Gather results from all processes
        audio_values = accelerator.gather(audio_values)

        if loggers:
            for logger in loggers:
                if hasattr(logger, "log_metrics"):
                    logger.log_metrics(
                        {"generated_samples": (batch_idx + 1) * batch_size},
                        step=batch_idx,
                    )

        # Only main process saves files
        if accelerator.is_main_process:
            for item_idx, audio in enumerate(audio_values):
                # Calculate global index accounting for distributed batches
                global_idx = (
                    batch_idx * batch_size * accelerator.num_processes
                    + item_idx
                )

                # Normalize/Clip audio to prevent artifacts
                audio_data = audio[0].float().cpu().numpy()
                audio_data = np.clip(audio_data, -1.0, 1.0)

                if global_idx < len(df):
                    sample_id = df.iloc[global_idx][id_column]
                    scipy.io.wavfile.write(
                        audio_dir / filename_template.format(sample_id),
                        sample_rate,
                        audio_data,
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
