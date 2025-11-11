from typing import Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from src.data.generic_data_module import GenericDataModule


class TTADataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.input_ids = input_ids.to(device)
        self.attention_mask = attention_mask.to(device)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[idx],
            self.attention_mask[idx]
        )


class TextDescriptions(GenericDataModule):
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        processor: AutoProcessor,
        emotions: list[str],
        instruments: list[str],
        genres: list[str],
        subset: str = "train",
        subset_size: float = 0.1,
        max_sequence_length: int = 256,
        device: torch.device = torch.device("cpu"),
    ):
        self.processor = processor
        self.max_sequence_length = max_sequence_length
        self.emotions = emotions
        self.instruments = instruments
        self.genres = genres
        self.subset = subset
        self.subset_size = subset_size
        super().__init__(dataset, batch_size, device)

    def _transform(self, dataset: pd.DataFrame):
        dataset = dataset.drop(
            columns=[
                # "start_s",
                # "end_s",
                # "audioset_positive_labels",
                # "author_id",
                # "is_balanced_subset",
                # "is_audioset_eval",
                "aspect_list"
            ]
        )

        # TODO: Extract concept tags

        return dataset

    def _tokenize(self, text: list[str]) -> dict[str, torch.Tensor]:
        inputs = self.processor(
            text=text,
            max_length=self.max_sequence_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        # {"input_ids": ..., "attention_mask": ...}
        return inputs

    def prepare_data(self):
        self.dataset_loaded = load_dataset(self.dataset)
        self.dataset = self._transform(
            self.dataset_loaded[self.subset].to_pandas()
        )
        self.dataset = self.dataset.sample(
            frac=self.subset_size
        ).reset_index(drop=True)

    def setup(self, stage=None):
        self.dataset_all = self._tokenize(
            self.dataset["caption"].tolist()
        )

    def random_dataloader(self):
        concept_dataset = TTADataset(
            input_ids=self.dataset_all["input_ids"],
            attention_mask=self.dataset_all["attention_mask"],
            device=self.device,
        )
        return DataLoader(
            dataset=concept_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
