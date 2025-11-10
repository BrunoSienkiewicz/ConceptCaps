from __future__ import annotations

from typing import Any, Dict


def flatten_numeric_metrics(value: Any, prefix: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, sub_value in value.items():
            new_prefix = f"{prefix}/{key}" if prefix else key
            result.update(flatten_numeric_metrics(sub_value, new_prefix))
    elif isinstance(value, (int, float)):
        if prefix:
            result[prefix] = float(value)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            new_prefix = f"{prefix}/{idx}" if prefix else str(idx)
            result.update(flatten_numeric_metrics(item, new_prefix))
    return result
