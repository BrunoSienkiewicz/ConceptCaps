from __future__ import annotations
import time

import torch
import hydra
import rootutils
import wandb
import lightning as pl

from transformers.trainer_utils import set_seed
from pathlib import Path

from src.caption import CaptionGenerationConfig, run_evaluation
from src.utils import print_config_tree, RankedLogger, instantiate_loggers
from src.caption.data import prepare_datasets
from src.caption.modeling import prepare_evaluation_model_tokenizer, prepare_tokenizer
from src.caption.evaluation import MetricComputer, run_test_evaluation
from datasets import load_dataset



rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)

@hydra.main(version_base=None, config_path="../../../config", config_name="caption_evaluation")
def main(cfg: CaptionGenerationConfig) -> None:
    log.info("Setting random seed...")
    set_seed(cfg.random_state)

    logger = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    dataset = load_dataset(cfg.data.dataset_name)
    dataset = prepare_datasets(cfg.data, cfg.prompt, dataset)
    test_examples = dataset["test"]

    log.info("Loading tokenizer...")
    tokenizer = prepare_tokenizer(cfg.model)

    metric_computer = MetricComputer(cfg.evaluation.metrics, tokenizer)

    data_dir = Path(cfg.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running evaluation...")
    log.info(f"Test examples count: {len(test_examples)}")
    model, tokenizer = prepare_evaluation_model_tokenizer(log, cfg.model)

    model.to(device)
    
    output_dir = Path(cfg.paths.data_dir) / cfg.model.name / cfg.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_test_evaluation(
        cfg, 
        metric_computer, 
        model, 
        tokenizer, 
        test_examples, 
        output_dir,
        log,
    )

    for l in logger:
        for key, value in metrics.items():
            l.log({key: value})

    return metrics

if __name__ == "__main__":
    main()