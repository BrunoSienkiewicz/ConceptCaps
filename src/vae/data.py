"""Data module for VAE training."""
import ast
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

from src.vae.config import VAEDataConfig, VAEConfig


class VAEDataModule(pl.LightningDataModule):
    """Lightning DataModule for VAE training on multi-label tag data."""
    
    def __init__(
        self,
        data_cfg: VAEDataConfig,
        taxonomy_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.05,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        
        # Load taxonomy
        with open(taxonomy_path, "r") as f:
            self.taxonomy = json.load(f)
        
        self._build_tag_mappings()
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def _build_tag_mappings(self):
        """Build mappings between tags and indices."""
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.cat_ranges = {}
        
        current_idx = 0
        categories = list(self.taxonomy.keys())
        
        for cat in categories:
            start = current_idx
            for tag in self.taxonomy[cat]:
                self.tag_to_idx[tag] = current_idx
                self.idx_to_tag[current_idx] = (cat, tag)
                current_idx += 1
            self.cat_ranges[cat] = (start, current_idx)
        
        self.total_input_dim = current_idx
    
    def setup(self, stage: Optional[str] = None):
        """Load and process data."""
        # Load dataset
        dataset = load_dataset(
            self.data_cfg.dataset_name,
            split=self.data_cfg.dataset_split
        )
        df = dataset.to_pandas()
        
        # Process data
        data = self._process_data_multilabel(df)
        
        # Split into train/val/test
        n_samples = len(data)
        n_test = int(n_samples * self.test_split)
        n_val = int((n_samples - n_test) * self.val_split)
        n_train = n_samples - n_val - n_test
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        self.train_data = torch.tensor(data[train_indices], dtype=torch.float32)
        self.val_data = torch.tensor(data[val_indices], dtype=torch.float32)
        self.test_data = torch.tensor(data[test_indices], dtype=torch.float32)
    
    def _process_data_multilabel(self, df: pd.DataFrame) -> np.ndarray:
        """
        Creates a Multi-Hot vector for every song.
        Example: [0, 1, 0, 1, 1, ...] where 1 means the tag is present.
        """
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                raw_tags = ast.literal_eval(row[self.data_cfg.aspect_column])
            except (ValueError, KeyError):
                continue
            
            raw_tags = [t.lower() for t in raw_tags]
            
            # Create Zero Vector
            vector = np.zeros(self.total_input_dim, dtype=np.float32)
            has_data = False
            
            for tag in raw_tags:
                if tag in self.tag_to_idx:
                    idx = self.tag_to_idx[tag]
                    vector[idx] = 1.0
                    has_data = True
            
            # Only keep records that have at least one valid tag
            if has_data:
                processed_data.append(vector)
        
        return np.array(processed_data)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=self.data_cfg.shuffle,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(self.val_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            TensorDataset(self.test_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
