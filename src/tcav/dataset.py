from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import MusicgenProcessor


class ConceptDataset(Dataset):
    def __init__(
        self,
        caption_column: str,
        df: pd.DataFrame,
        concept_tensor: torch.Tensor,
        processor: MusicgenProcessor,
        device: str = "cpu",
    ):
        self.df = df
        self.caption_column = caption_column
        self.concept_tensor = concept_tensor.to(device)
        self.text = self.df[self.caption_column].tolist()

        inputs = processor(
            text=self.text,
            max_length=256,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.concept_tensor
