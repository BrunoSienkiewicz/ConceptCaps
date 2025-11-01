from __future__ import annotations

from typing import Any, Dict, List, Tuple

from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig


def _format_prompt(prompt_cfg: DictConfig, aspects: Any, reference_caption: str) -> str:
    user_prompt = prompt_cfg["user_prompt_template"].format(tags=aspects)
    return prompt_cfg["template"].format(
        system_prompt=prompt_cfg["system_prompt"],
        user_prompt=user_prompt,
        assistant_response=reference_caption,
    ).strip()


def prepare_datasets(data_cfg: DictConfig) -> Tuple[DatasetDict, List[dict]]:
    data_files = {
        "train": data_cfg.train_file,
        "validation": data_cfg.validation_file,
        "test": data_cfg.test_file,
    }
    raw_dataset = load_dataset("csv", data_files=data_files)

    test_dataset_raw = raw_dataset["test"]
    prompt_cfg = data_cfg.prompt
    text_column = data_cfg.text_column
    remove_columns = data_cfg.remove_columns
    if remove_columns is None:
        remove_columns = raw_dataset["train"].column_names

    def _transform_row(row: Dict[str, Any]) -> Dict[str, str]:
        formatted = _format_prompt(
            prompt_cfg,
            row[data_cfg.aspect_column],
            row[data_cfg.caption_column],
        )
        return {text_column: formatted}

    processed_dataset = raw_dataset.map(
        _transform_row,
        remove_columns=remove_columns,
    )

    return processed_dataset, test_dataset_raw.to_list()
