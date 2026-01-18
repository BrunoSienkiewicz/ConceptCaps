"""Simplified TTA dataset loader.

Direct dataset loading and tokenization without unnecessary abstractions.
"""

from typing import Tuple

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor


class TTADataset(Dataset):
    """Simple TTA dataset wrapper."""

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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.input_ids[idx], self.attention_mask[idx])


def load_and_tokenize_dataset(
    dataset_name: str,
    processor: AutoProcessor,
    subset: str = "train",
    subset_size: float = 0.1,
    max_sequence_length: int = 256,
    caption_column: str = "caption",
    device: torch.device = torch.device("cpu"),
) -> Tuple[TTADataset, pd.DataFrame]:
    """Load a dataset and tokenize captions.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        processor: Tokenizer/processor to use
        subset: Dataset split to load
        subset_size: Fraction of dataset to use
        max_sequence_length: Max tokenization length
        caption_column: Name of caption column
        device: Device to load tensors to
        
    Returns:
        (TTADataset, metadata_dataframe)
    """
    # Load dataset
    dataset_dict = load_dataset(dataset_name)
    df = dataset_dict[subset].to_pandas()
    
    # Sample subset
    df = df.sample(frac=subset_size).reset_index(drop=True)
    
    # Tokenize captions
    captions = df[caption_column].tolist()
    inputs = processor(
        text=captions,
        max_length=max_sequence_length,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )
    
    # Create dataset
    tta_dataset = TTADataset(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        device=device,
    )
    
    return tta_dataset, df

def prepare_dataloader(cfg: dict, processor: AutoProcessor) -> Tuple[DataLoader, pd.DataFrame]:
    """Prepare DataLoader for TTA dataset.
    
    Args:
        cfg: Configuration dictionary with dataset and dataloader parameters.
    Returns:
        DataLoader for the TTA dataset.
    """
    dataset, df = load_and_tokenize_dataset(
        dataset_name=cfg.get("dataset_name", "your/default-dataset"),
        processor=processor,
        subset=cfg.get("subset", "train"),
        subset_size=cfg.get("subset_size", 0.1),
        max_sequence_length=cfg.get("max_sequence_length", 256),
        caption_column=cfg.get("text_column", "caption"),
        device=torch.device(cfg.get("device", "cpu")),
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=cfg.get("shuffle", False),
    )
    return dataloader, df


def save_dataframe_metadata(
    df: pd.DataFrame,
    data_dir: str,
    id_column: str = "id",
    filename_template: str = "{}.wav",
) -> None:
    """Save dataframe metadata with audio file paths.
    
    Args:
        df: DataFrame containing metadata
        data_dir: Directory where audio files are stored
        id_column: Column in df that contains unique IDs
        filename_template: Template for audio filenames
    """
    df["filename"] = df[id_column].apply(lambda x: filename_template.format(x))
    df.to_csv(f"{data_dir}/metadata.csv", index=False)
