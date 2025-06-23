from pathlib import Path

import hydra
import matplotlib
import pandas as pd
import pytorch_lightning as pl
import rootutils
import torch
import wandb
from captum.concept import TCAV
from captum.concept._utils.common import concepts_to_str

from src.tcav.concept import create_experimental_set
from src.tcav.config import TCAVConfig
from src.utils import (RankedLogger, instantiate_loggers, log_hyperparameters,
                       print_config_tree)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)


def format_float(f):
    return float(f"{f:.3f}" if abs(f) >= 0.0005 else f"{f:.3e}")


def save_results(experimental_sets, tcav_scores):
    df_scores = []
    df_concepts = []
    layers = []
    sets = []

    for idx_es, concepts in enumerate(experimental_sets):
        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)
        for i in range(len(concepts)):
            for layer, scores in tcav_scores[concepts_key].items():
                val = format_float(scores["sign_count"][i])
                df_scores.append(val)
                df_concepts.append(concepts[i].name)
                layers.append(layer)
                sets.append(str(idx_es))
    res_df = pd.DataFrame.from_dict(
        {
            "Scores": df_scores,
            "Concept": df_concepts,
            "Layer": layers,
            "Set": sets,
        }
    )

    res_df["Scores"] = res_df["Scores"].astype(float)
    res_df["Set"] = res_df["Set"].astype(int)
    res_df["Layer"] = res_df["Layer"].astype(str)
    res_df["Concept"] = res_df["Concept"].astype(str)
    return res_df


def tcav(cfg: TCAVConfig):
    random_state = cfg.random_state
    pl.seed_everything(random_state)

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))
    matplotlib.use(cfg.experiment.plot_backend)
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
    classifier = hydra.utils.instantiate(cfg.model.classifier, trainer=trainer)

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
        # attribute_to_layer_input=True # Hardcoded for now, can be extended later
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
    log.info(f"Saving outputs to {cfg.paths.output_dir}")

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tcav_scores_df = save_results(experimental_set, tcav_scores)
    tcav_scores_df.to_csv(output_dir / "tcav.csv", index=False)


@hydra.main(version_base=None, config_path="../../config", config_name="tcav")
def main(cfg: TCAVConfig):
    print_config_tree(cfg)
    tcav(cfg)


if __name__ == "__main__":
    main()
