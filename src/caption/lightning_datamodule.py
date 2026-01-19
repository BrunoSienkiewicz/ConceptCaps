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
        prompt_cfg: DictConfig,
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 512,
    ):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.data_cfg = data_cfg
        self.prompt_cfg = prompt_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

    def tokenize_function(self, examples):
        text_column = self.data_cfg.text_column
        _examples = [
            text + self.tokenizer.eos_token if not text.endswith(self.tokenizer.eos_token) else text
            for text in examples[text_column]
        ]
        
        tokenized = self.tokenizer(
            _examples,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        
        prompt_delimiter = self.prompt_cfg.prompt_delimiter.strip()
        delimiter_token_ids = self.tokenizer.encode(prompt_delimiter, add_special_tokens=False)
        
        labels = []
        for token_ids in tokenized["input_ids"]:
            label = token_ids.copy() if hasattr(token_ids, 'copy') else list(token_ids)
            token_list = label if isinstance(label, list) else label.tolist()
            
            marker_found = False
            for j in range(len(token_list) - len(delimiter_token_ids) + 1):
                if token_list[j:j+len(delimiter_token_ids)] == delimiter_token_ids:
                    mask_until = j + len(delimiter_token_ids)
                    label[:mask_until] = [-100] * mask_until
                    marker_found = True
                    break
            
            if not marker_found:
                label[:] = [-100] * len(label)
                log.warning(f"Prompt delimiter not found in example")
            
            labels.append(label)
        
        tokenized["labels"] = labels
            
        return tokenized

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset["train"].map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.dataset["train"].column_names,
                desc="Tokenizing train dataset",
            )
            self.val_dataset = self.dataset["validation"].map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.dataset["validation"].column_names,
                desc="Tokenizing validation dataset",
            )
    
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset["test"].map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.dataset["test"].column_names,
                desc="Tokenizing test dataset",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    def test_dataloader(self):
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
