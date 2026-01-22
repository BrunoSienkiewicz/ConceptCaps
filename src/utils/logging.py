from pathlib import Path
from typing import Any, Dict, List, Sequence

import hydra
import rich
import rich.syntax
import rich.tree
from lightning import Callback
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback
        configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich
    library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        if field not in cfg:
            log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."  # noqa
            )
        else:
            queue.append(field)

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)
