from __future__ import annotations

import os

from pathlib import Path

import scipy.io
import torch
import pandas as pd
from torch.utils.data import DataLoader


def generate_audio_samples(
    model,
    dataloader: DataLoader,
    audio_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    df: pd.DataFrame,
    id_column: str = "id",
    filename_template: str = "{}.wav",
) -> None:
    os.makedirs(audio_dir, exist_ok=True)
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
        sampling_rate = model.config.audio_encoder.sampling_rate
        for item_idx, audio in enumerate(audio_values):
            global_idx = batch_idx * batch_size + item_idx
            sample_id = df.iloc[global_idx][id_column]
            scipy.io.wavfile.write(
                audio_dir / filename_template.format(sample_id),
                sampling_rate,
                audio[0].cpu().numpy(),
            )
