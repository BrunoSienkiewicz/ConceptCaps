from __future__ import annotations

from pathlib import Path

import scipy.io
import torch
from torch.utils.data import DataLoader
from transformers import MusicGenModel


def generate_audio_samples(
    model: MusicGenModel,
    dataloader: DataLoader,
    audio_dir: Path,
    max_new_tokens: int,
    batch_size: int,
) -> None:
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        with torch.no_grad():
            audio_values = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        sampling_rate = model.model.config.audio_encoder.sampling_rate
        for item_idx, audio in enumerate(audio_values):
            global_idx = batch_idx * batch_size + item_idx
            scipy.io.wavfile.write(
                audio_dir / f"{global_idx}.wav",
                sampling_rate,
                audio[0].cpu().numpy(),
            )
