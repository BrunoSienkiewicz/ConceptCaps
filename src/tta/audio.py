from __future__ import annotations

import os

from pathlib import Path

import scipy.io
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator, PartialState


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
) -> None:
    os.makedirs(audio_dir, exist_ok=True)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating audio")):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        
        # Use inference mode for better performance
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
        
        sampling_rate = model.config.audio_encoder.sampling_rate
        for item_idx, audio in enumerate(audio_values):
            global_idx = batch_idx * batch_size + item_idx
            sample_id = df.iloc[global_idx][id_column]
            scipy.io.wavfile.write(
                audio_dir / filename_template.format(sample_id),
                sampling_rate,
                audio[0].float().cpu().numpy(),
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
) -> None:
    """Generate audio using Accelerate for multi-GPU distribution."""
    
    accelerator = Accelerator()
    
    os.makedirs(audio_dir, exist_ok=True)
    
    # Prepare model and dataloader with accelerate
    model, dataloader = accelerator.prepare(model, dataloader)
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if guidance_scale is not None:
        generation_kwargs["guidance_scale"] = guidance_scale
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating audio", disable=not accelerator.is_local_main_process)):
        input_ids, attention_mask = batch
        
        with torch.inference_mode():
            audio_values = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        
        # Gather results from all processes
        audio_values = accelerator.gather(audio_values)
        
        if accelerator.is_main_process:
            for item_idx, audio in enumerate(audio_values):
                global_idx = batch_idx * batch_size * accelerator.num_processes + item_idx
                if global_idx < len(df):
                    sample_id = df.iloc[global_idx][id_column]
                    scipy.io.wavfile.write(
                        audio_dir / filename_template.format(sample_id),
                        sampling_rate,
                        audio[0].float().cpu().numpy(),
                    )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()