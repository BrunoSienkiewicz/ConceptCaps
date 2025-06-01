from functools import reduce
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rootutils
import torch
import wandb
from captum.concept import TCAV
from captum.concept._utils.common import concepts_to_str

from src.tcav.concept import create_experimental_set
from src.tcav.config import TCAVConfig
from src.utils import (
    RankedLogger,
    instantiate_loggers,
    log_hyperparameters,
    print_config_tree,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)


def format_float(f):
    return float(f"{f:.3f}" if abs(f) >= 0.0005 else f"{f:.3e}")


def plot_tcav_scores(experimental_sets, tcav_scores, layers):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize=(25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i - 1]])
        _ax = ax[idx_es] if len(experimental_sets) > 1 else ax
        for i in range(len(concepts)):
            val = [
                format_float(scores["sign_count"][i])
                for layer, scores in tcav_scores[concepts_key].items()
            ]
            _ax.bar(
                pos[i],
                val,
                width=barWidth,
                edgecolor="white",
                label=concepts[i].name,
            )

        _ax.set_xlabel(f"Set {str(idx_es)}", fontweight="bold", fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16, rotation=45)
        _ax.legend(fontsize=16)

    plt.tight_layout()


def tcav(cfg: TCAVConfig):
    random_state = cfg.random_state
    pl.seed_everything(random_state)

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))
    wandb.login()

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model.model._target_}>")
    model = hydra.utils.instantiate(cfg.model.model)
    model.to(device)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=None,  # No callbacks needed for TCAV
        enable_progress_bar=False,  # Disable progress bar for cleaner output
    )

    log.info(f"Instantiating classifier <{cfg.model.classifier._target_}>")
    classifier = hydra.utils.instantiate(cfg.model.classifier)

    object_dict = {
        "cfg": cfg,
        "data_module": data_module,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "classifier": classifier,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

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

    instrument_tcav = TCAV(
        model=model,
        model_id=cfg.model.model_id,
        classifier=classifier,
        layer_attr_method=hydra.utils.instantiate(
            cfg.experiment.layer_attr_method, model.forward, None
        ),
        layers=layers,
        show_progress=True,
        save_path=cfg.paths.output_dir,
    )

    if cfg.experiment.n_groups != 0:
        layer_masks = None
    else:
        # For now I am creating the layer masks by grouping adjacent neurons
        # into groups of size n_groups. Later we can add a more sophisticated
        # approach to create the layer masks.
        n_groups = cfg.experiment.n_groups
        layer_masks = []
        for layer in layers:
            layer_shape = reduce(getattr, layer.split("."), model).weight.shape[0]
            layer_mask = torch.zeros(layer_shape).to(device)
            group_size = layer_mask.shape[0] // n_groups
            for i in range(n_groups + 1):
                layer_mask[i * group_size : (i + 1) * group_size] = i
            layer_masks.append(layer_mask)

        layer_masks = (*layer_masks,)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    log.info("Running TCAV...")
    tcav_scores = instrument_tcav.interpret(
        inputs=(input_ids, attention_mask, None),
        experimental_sets=experimental_set,
        target=0,
        layer_mask=layer_masks,
    )

    log.info("TCAV scores computed successfully!")
    plot_tcav_scores(experimental_set, tcav_scores, layers)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tcav_scores_df = pd.DataFrame.from_dict(tcav_scores, orient="index")
    tcav_scores_df.to_csv(output_dir / "tcav.csv", index=False)
    plt.savefig(output_dir / "tcav_scores.png", bbox_inches="tight", dpi=300)


@hydra.main(version_base=None, config_path="../../config", config_name="tcav")
def main(cfg: TCAVConfig):
    print_config_tree(cfg)
    tcav(cfg)


if __name__ == "__main__":
    main()
