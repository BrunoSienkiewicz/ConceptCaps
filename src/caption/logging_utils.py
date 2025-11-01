from __future__ import annotations

from typing import Any, Dict

import wandb
from transformers.trainer_callback import TrainerCallback


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


class WandbMonitoringCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs) -> None:  # type: ignore[override]
        if logs is None or wandb.run is None:
            return
        payload: Dict[str, float] = {}
        for key, value in logs.items():
            if key == "epoch":
                continue
            if key.startswith("eval_"):
                base = f"eval/{key[5:]}"
            elif key.startswith("train_"):
                base = f"train/{key[6:]}"
            else:
                base = f"train/{key}"
            payload.update(flatten_numeric_metrics(value, base))
        if state.global_step is not None:
            payload["trainer/global_step"] = state.global_step
        if state.epoch is not None:
            payload["trainer/epoch"] = state.epoch
        if payload:
            wandb.log(payload, step=state.global_step)
