from __future__ import annotations

import os

from typing import Any, Dict, List, Tuple
from pathlib import Path

from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig


def _format_prompt(prompt_cfg: DictConfig, aspects: Any, reference_caption: str) -> str:
    user_prompt = prompt_cfg["user_prompt_template"].format(tags=aspects)
    return prompt_cfg["template"].format(
        system_prompt=prompt_cfg["system_prompt"],
        user_prompt=user_prompt,
        assistant_response=reference_caption,
    ).strip()

    
def _format_eval_prompt(prompt_cfg: DictConfig, aspects: Any) -> str:
    user_prompt = prompt_cfg["user_prompt_template"].format(tags=aspects)
    return prompt_cfg["eval_template"].format(
        system_prompt=prompt_cfg["system_prompt"],
        user_prompt=user_prompt,
    ).strip()


def prepare_datasets(data_cfg, prompt_cfg, raw_dataset: DatasetDict) -> DatasetDict:
    text_column = data_cfg.text_column
    caption_column = data_cfg.caption_column
    aspect_column = data_cfg.aspect_column

    def _transform_train_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = _format_prompt(
            prompt_cfg,
            row[aspect_column],
            row[caption_column],
        )
        return {text_column: formatted}

    def _transform_eval_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = _format_eval_prompt(
            prompt_cfg,
            row[data_cfg.aspect_column],
        )
        return {
            text_column: formatted, 
            caption_column: row[caption_column],
            aspect_column: row[aspect_column],
        }

    processed_dataset = DatasetDict()
    for split in raw_dataset.keys():
        if split == "train":
            processed_dataset[split] = raw_dataset[split].map(
                _transform_train_row,
            )
        else:
            processed_dataset[split] = raw_dataset[split].map(
                _transform_eval_row,
            )

    return processed_dataset


def prepare_inference_datasets(
    data_cfg,
    prompt_cfg,
    raw_dataset: DatasetDict,
) -> DatasetDict:
    text_column = data_cfg.text_column
    aspect_column = data_cfg.aspect_column
    id_column = data_cfg.id_column
    remove_columns = data_cfg.remove_columns
    if remove_columns is None:
        remove_columns = raw_dataset["test"].column_names

    def _transform_inference_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = _format_eval_prompt(
            prompt_cfg,
            row[aspect_column],
        )
        return {
            text_column: formatted,
            aspect_column: row[aspect_column],
            id_column: row[id_column],
        }

    processed_dataset = DatasetDict()
    for split in raw_dataset.keys():
        processed_dataset[split] = raw_dataset[split].map(
            _transform_inference_row,
            remove_columns=remove_columns,
        )

    return processed_dataset