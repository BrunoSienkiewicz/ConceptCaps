"""
Module for loading and processing MTG Jamendo dataset for caption generation.

This module extends the existing data loading functionality to work with
the MTG Jamendo dataset that has pre-categorized tags (instrument, mood, genre, tempo).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_dataset, Dataset
from omegaconf import DictConfig


def load_mtg_jamendo_dataset(data_cfg: DictConfig) -> DatasetDict:
    """
    Load MTG Jamendo dataset from CSV files.

    Args:
        data_cfg: Data configuration

    Returns:
        DatasetDict with train, validation, and test splits
    """
    datasets = {}

    for split_name, file_path in [
        ("train", data_cfg.train_file),
        ("validation", data_cfg.validation_file),
        ("test", data_cfg.test_file),
    ]:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Ensure all expected columns exist
            required_cols = ["caption", "instrument_tags", "mood_tags", "genre_tags", "tempo_tags"]
            for col in required_cols:
                if col not in df.columns:
                    if col == "caption" and "description" in df.columns:
                        df.rename(columns={"description": "caption"}, inplace=True)
                    else:
                        df[col] = ""

            # Create combined aspect string for compatibility
            df["aspect_list"] = (
                df["instrument_tags"].fillna("") + ", " +
                df["mood_tags"].fillna("") + ", " +
                df["genre_tags"].fillna("") + ", " +
                df["tempo_tags"].fillna("")
            ).str.replace(", ,", ",").str.strip(", ")

            # Create HuggingFace dataset
            datasets[split_name] = Dataset.from_pandas(df)
        else:
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return DatasetDict(datasets)


def format_mtg_jamendo_prompt(
    prompt_cfg: DictConfig,
    instrument_tags: str,
    mood_tags: str,
    genre_tags: str,
    tempo_tags: str,
    caption: str = "",
) -> str:
    """
    Format a prompt with categorized tags.

    Args:
        prompt_cfg: Prompt configuration
        instrument_tags: Comma-separated instrument tags
        mood_tags: Comma-separated mood tags
        genre_tags: Comma-separated genre tags
        tempo_tags: Comma-separated tempo tags
        caption: Reference caption (for training)

    Returns:
        Formatted prompt string
    """
    template = prompt_cfg.get("user_prompt_template", "")

    user_prompt = template.format(
        instruments=instrument_tags,
        moods=mood_tags,
        genres=genre_tags,
        tempos=tempo_tags,
    )

    if caption:
        return prompt_cfg["template"].format(
            system_prompt=prompt_cfg["system_prompt"],
            user_prompt=user_prompt,
            assistant_response=caption,
        ).strip()
    else:
        return prompt_cfg.get("eval_template", prompt_cfg["template"]).format(
            system_prompt=prompt_cfg["system_prompt"],
            user_prompt=user_prompt,
            assistant_response="",
        ).strip()


def prepare_mtg_jamendo_datasets(data_cfg, raw_dataset: DatasetDict) -> DatasetDict:
    """
    Prepare MTG Jamendo datasets for fine-tuning.

    Args:
        data_cfg: Data configuration
        raw_dataset: Raw dataset from load_mtg_jamendo_dataset

    Returns:
        Processed DatasetDict ready for training
    """
    prompt_cfg = data_cfg.prompt
    text_column = data_cfg.text_column
    caption_column = data_cfg.caption_column
    remove_columns = data_cfg.get("remove_columns", None)

    if remove_columns is None:
        remove_columns = raw_dataset["train"].column_names

    def _transform_train_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = format_mtg_jamendo_prompt(
            prompt_cfg,
            row.get("instrument_tags", ""),
            row.get("mood_tags", ""),
            row.get("genre_tags", ""),
            row.get("tempo_tags", ""),
            row.get(caption_column, ""),
        )
        return {text_column: formatted}

    def _transform_eval_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = format_mtg_jamendo_prompt(
            prompt_cfg,
            row.get("instrument_tags", ""),
            row.get("mood_tags", ""),
            row.get("genre_tags", ""),
            row.get("tempo_tags", ""),
        )
        return {
            text_column: formatted,
            caption_column: row.get(caption_column, ""),
            "instrument_tags": row.get("instrument_tags", ""),
            "mood_tags": row.get("mood_tags", ""),
            "genre_tags": row.get("genre_tags", ""),
            "tempo_tags": row.get("tempo_tags", ""),
        }

    processed_dataset = DatasetDict()

    for split in raw_dataset.keys():
        if split == "train":
            processed_dataset[split] = raw_dataset[split].map(
                _transform_train_row,
                remove_columns=remove_columns,
            )
            if data_cfg.get("max_train_samples") is not None:
                processed_dataset[split] = processed_dataset[split].select(
                    range(data_cfg.max_train_samples)
                )
        else:
            processed_dataset[split] = raw_dataset[split].map(
                _transform_eval_row,
                remove_columns=remove_columns,
            )
            if data_cfg.get("max_eval_samples") is not None:
                processed_dataset[split] = processed_dataset[split].select(
                    range(data_cfg.max_eval_samples)
                )

    return processed_dataset
