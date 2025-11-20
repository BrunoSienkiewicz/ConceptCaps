from __future__ import annotations

from typing import Optional

import lightning as pl
from datasets import DatasetDict
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class CaptionDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for caption fine-tuning."""

    def __init__(
        self,
        dataset: DatasetDict,
        tokenizer: AutoTokenizer,
        data_cfg: DictConfig,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 512,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        
        # Data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

    def tokenize_function(self, examples):
        """Tokenize the text column."""
        text_column = self.data_cfg.text_column
        # Add EOS token if not present
        _examples = [
            text + self.tokenizer.eos_token if not text.endswith(self.tokenizer.eos_token) else text
            for text in examples[text_column]
        ]
        
        # Tokenize
        tokenized = self.tokenizer(
            _examples,
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        
        # For training, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/test."""
        if stage == "fit" or stage is None:
            # Tokenize training dataset
            if "train" in self.dataset:
                self.train_dataset = self.dataset["train"].map(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=self.dataset["train"].column_names,
                    desc="Tokenizing training dataset",
                )
            
            # Tokenize validation dataset
            if "validation" in self.dataset:
                self.val_dataset = self.dataset["validation"].map(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=self.dataset["validation"].column_names,
                    desc="Tokenizing validation dataset",
                )
        
        if stage == "test" or stage is None:
            # Tokenize test dataset
            if "test" in self.dataset:
                self.test_dataset = self.dataset["test"].map(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=self.dataset["test"].column_names,
                    desc="Tokenizing test dataset",
                )

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return test dataloader."""
        if hasattr(self, "test_dataset"):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.data_collator,
                pin_memory=True,
            )
        return None
