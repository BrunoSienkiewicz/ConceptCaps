from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import pytorch_lightning as pl
import torch
import wandb
from captum.concept import TCAV

from src.tcav.concept import create_experimental_set
from src.tcav.config import TCAVConfig
from src.tcav.results import build_results_dataframe
from src.utils import RankedLogger, instantiate_loggers, log_hyperparameters


def run_tcav(cfg: TCAVConfig) -> Path:
    log = RankedLogger(__name__, rank_zero_only=True)

    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    experiment_loggers = instantiate_loggers(cfg.get("logger"))
    matplotlib.use(cfg.experiment.plot_backend)
    wandb.login()

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model.model._target_}>")
    model = hydra.utils.instantiate(cfg.model.model)
    model.to(device)

    log.info(f"Instantiating classifier <{cfg.model.classifier._target_}>")
    classifier = hydra.utils.instantiate(cfg.model.classifier)

    if experiment_loggers:
        log.info("Logging hyperparameters...")
        log_hyperparameters(
            {
                "cfg": cfg,
                "data_module": data_module,
                "model": model,
                "logger": experiment_loggers,
                "classifier": classifier,
            }
        )

    log.info("Preparing data...")
    data_module.prepare_data()
    data_module.setup()

    log.info("Creating experimental set...")
    experimental_set = create_experimental_set(
        data_module,
        experimental_set_size=cfg.experiment.experimental_set_size,
        num_samples=cfg.experiment.num_samples,
        device=device,
    )
    inputs = data_module.select_samples(
        num_samples=cfg.experiment.num_samples,
    )

    layers = cfg.experiment.layers

    tcav = TCAV(
        model=model,
        model_id=cfg.model.model_id,
        classifier=classifier,
        layers=layers,
        show_progress=True,
        save_path=cfg.paths.output_dir,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    log.info("Running TCAV...")
    tcav_scores = hydra.utils.instantiate(
        cfg.experiment.interpretation_method,
        tcav=tcav,
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        experimental_set=experimental_set,
        layers=layers,
        device=device,
    )

    log.info("TCAV scores computed successfully!")
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving outputs to {output_dir}")
    tcav_scores_df = build_results_dataframe(experimental_set, tcav_scores)
    output_path = output_dir / "tcav.csv"
    tcav_scores_df.to_csv(output_path, index=False)

    return output_path
