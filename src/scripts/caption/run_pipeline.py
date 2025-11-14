from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import hydra
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def _ensure_list(value: Sequence[str] | None) -> list[str]:
    return list(value) if value else []


def _build_stage_command(
    stage: DictConfig,
    root_dir: Path,
    runtime_dir: Path,
    base_overrides: Sequence[str],
) -> tuple[list[str], Path]:
    stage_dir = runtime_dir / stage.name
    stage_dir.mkdir(parents=True, exist_ok=True)

    overrides = [*base_overrides, *(_ensure_list(stage.get("overrides")))]
    overrides.append(f"hydra.run.dir={stage_dir}")

    script_path = root_dir / stage.script
    if not script_path.exists():
        raise FileNotFoundError(f"Stage script not found: {script_path}")

    command = [sys.executable, str(script_path), *overrides]
    return command, stage_dir


def _run_stage(command: list[str], env: dict[str, str]) -> float:
    log.info("Executing: %s", " ".join(command))
    start = time.monotonic()
    subprocess.run(command, check=True, env=env)
    return time.monotonic() - start


def _prepare_env(cfg: DictConfig, root_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PROJECT_ROOT"] = str(root_dir)
    env["OUT_DIR"] = str(Path(cfg.storage.out_dir))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root_dir), env.get("PYTHONPATH", "")]))
    env["HF_HOME"] = str(Path(cfg.storage.cache_dir))
    Path(cfg.storage.cache_dir).mkdir(parents=True, exist_ok=True)
    log.info("Environment prepared with PROJECT_ROOT=%s and HF_HOME=%s", root_dir, cfg.storage.cache_dir)
    return env


def _finalize_dataset(
    inference_stage_dir: Path,
    ready_dir: Path,
    run_id: str,
) -> list[Path]:
    if not inference_stage_dir.exists():
        log.warning("Inference stage output not found at %s", inference_stage_dir)
        return []

    destination = ready_dir / run_id
    destination.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for candidate in inference_stage_dir.glob("*.csv"):
        dest = destination / candidate.name
        shutil.copy(candidate, dest)
        copied.append(dest)
        log.info("Copied inference artifact %s to %s", candidate, dest)

    if not copied:
        log.warning("No inference artifacts found inside %s", inference_stage_dir)

    return copied


@hydra.main(version_base=None, config_path="../../../config", config_name="caption_pipeline")
def main(cfg: DictConfig) -> None:
    root_dir = Path(cfg.paths.root_dir).expanduser()
    pipeline_root = Path(cfg.dataset.base_dir).expanduser()
    pipeline_root.mkdir(parents=True, exist_ok=True)

    run_dir = pipeline_root / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _prepare_env(cfg, root_dir)

    base_overrides = _ensure_list(cfg.pipeline.base_overrides)
    stage_summaries: list[dict[str, str | float]] = []

    for stage_cfg in cfg.pipeline.stages:
        command, stage_dir = _build_stage_command(stage_cfg, root_dir, run_dir, base_overrides)
        duration = _run_stage(command, env)
        stage_summaries.append({
            "name": stage_cfg.name,
            "duration_seconds": duration,
            "output_dir": str(stage_dir),
        })

    inference_dir = run_dir / cfg.pipeline.inference_stage
    dataset_files = _finalize_dataset(inference_dir, Path(cfg.dataset.ready_dir), cfg.run_id)

    summary = {
        "run_id": cfg.run_id,
        "stages": stage_summaries,
        "dataset": {
            "ready_dir": str(Path(cfg.dataset.ready_dir) / cfg.run_id),
            "files": [str(p) for p in dataset_files],
        },
    }

    summary_path = run_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Pipeline finished; summary written to %s", summary_path)


if __name__ == "__main__":
    main()
