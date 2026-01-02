from __future__ import annotations

import os
from pathlib import Path

import scipy.io
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


class DataParallelGeneration(nn.DataParallel):
    """
    DataParallel wrapper that properly handles generation methods.
    Fixes the issue where DataParallel doesn't forward generation kwargs correctly.
    """
    def generate(self, **kwargs):
        return self.module.generate(**kwargs)
    
    @property
    def config(self):
        return self.module.config


def generate_audio_samples_parallel(
    model,
    dataloader: DataLoader,
    audio_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    df: pd.DataFrame,
    device_ids: list[int] | None = None,
    id_column: str = "id",
    filename_template: str = "{}.wav",
) -> None:
    """
    Generate audio samples using multiple GPUs with DataParallel.
    
    Args:
        model: The model to use for generation
        dataloader: DataLoader providing batches
        audio_dir: Directory to save generated audio files
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size (per-GPU batch size will be batch_size // num_gpus)
        df: DataFrame with metadata
        device_ids: List of GPU IDs to use (e.g., [0, 1]). If None, uses all available GPUs.
        id_column: Column name for sample IDs
        filename_template: Template for output filenames
    """
    os.makedirs(audio_dir, exist_ok=True)
    
    # Determine device IDs
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    num_gpus = len(device_ids)
    print(f"Using {num_gpus} GPUs: {device_ids}")
    print(f"Total batch size: {batch_size}, Per-GPU batch size: {batch_size // num_gpus}")
    
    # Wrap model in DataParallel
    if num_gpus > 1:
        model = DataParallelGeneration(model, device_ids=device_ids)
        primary_device = f"cuda:{device_ids[0]}"
    else:
        primary_device = model.device
    
    model.eval()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating audio")):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(primary_device)
        attention_mask = attention_mask.to(primary_device)
        
        with torch.no_grad():
            audio_values = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        
        # Get sampling rate from the model config
        if isinstance(model, DataParallelGeneration):
            sampling_rate = model.module.config.audio_encoder.sampling_rate
        else:
            sampling_rate = model.config.audio_encoder.sampling_rate
        
        # Save each audio file
        for item_idx, audio in enumerate(audio_values):
            global_idx = batch_idx * batch_size + item_idx
            if global_idx >= len(df):
                break
            sample_id = df.iloc[global_idx][id_column]
            scipy.io.wavfile.write(
                audio_dir / filename_template.format(sample_id),
                sampling_rate,
                audio[0].cpu().numpy(),
            )


def generate_audio_samples_manual_split(
    model,
    dataloader: DataLoader,
    audio_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    df: pd.DataFrame,
    device_ids: list[int] = [0, 1],
    id_column: str = "id",
    filename_template: str = "{}.wav",
) -> None:
    """
    Alternative approach: Manually split batches across GPUs.
    This gives more control but is more complex.
    
    Use this if DataParallel has issues with the generation method.
    """
    os.makedirs(audio_dir, exist_ok=True)
    
    num_gpus = len(device_ids)
    print(f"Using {num_gpus} GPUs with manual splitting: {device_ids}")
    
    # Clone model to each GPU
    models = []
    for device_id in device_ids:
        device = f"cuda:{device_id}"
        model_copy = model.to(device) if device_id == device_ids[0] else model
        models.append((device, model_copy))
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating audio")):
        input_ids, attention_mask = batch
        
        # Split batch across GPUs
        batch_per_gpu = input_ids.size(0) // num_gpus
        audio_outputs = []
        
        for gpu_idx, (device, gpu_model) in enumerate(models):
            start_idx = gpu_idx * batch_per_gpu
            end_idx = start_idx + batch_per_gpu if gpu_idx < num_gpus - 1 else input_ids.size(0)
            
            gpu_input_ids = input_ids[start_idx:end_idx].to(device)
            gpu_attention_mask = attention_mask[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                gpu_audio = gpu_model.generate(
                    input_ids=gpu_input_ids,
                    attention_mask=gpu_attention_mask,
                    max_new_tokens=max_new_tokens,
                )
            audio_outputs.append(gpu_audio)
        
        # Combine outputs
        audio_values = torch.cat([a.cpu() for a in audio_outputs], dim=0)
        
        sampling_rate = model.config.audio_encoder.sampling_rate
        
        # Save each audio file
        for item_idx, audio in enumerate(audio_values):
            global_idx = batch_idx * batch_size + item_idx
            if global_idx >= len(df):
                break
            sample_id = df.iloc[global_idx][id_column]
            scipy.io.wavfile.write(
                audio_dir / filename_template.format(sample_id),
                sampling_rate,
                audio[0].numpy(),
            )
