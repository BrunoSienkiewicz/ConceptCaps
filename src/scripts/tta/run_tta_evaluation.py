from pathlib import Path

import hydra
import rootutils
import torch
import pytorch_lightning as pl

from src.utils import (RankedLogger, instantiate_loggers,
                       print_config_tree)
from src.tta.config import TTAConfig
from tta.evaluation import TTAEvaluator


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tta_generation")
def main(cfg: TTAConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    log.info("Setting random seed...")
    pl.seed_everything(cfg.random_state)

    # Setup loggers
    loggers = []
    if cfg.get("logger"):
        pl_loggers = instantiate_loggers(cfg.logger)
        if pl_loggers:
            loggers.extend(pl_loggers if isinstance(pl_loggers, list) else [pl_loggers])

    print_config_tree(cfg)

    data_dir = Path(cfg.evaluation.get("data_dir", Path.cwd() / "tta_evaluation"))

    log.info("Initializing TTA evaluator...")
    evaluator = TTAEvaluator(
        clap_model=cfg.evaluation.get("clap_model", "laion/clap-htsat-unfused"),
        device=str(device),
    )

    log.info("Running TTA evaluation...")
    results = evaluator.evaluate(
        generated_audio_dir=data_dir / "audio_samples",
        metadata_path=data_dir / "metadata.csv",
        reference_audio_dir=cfg.evaluation.get("reference_audio_dir", data_dir / "reference_audio_samples"),
        output_dir=data_dir / "evaluation_results",
        text_column=cfg.data.get("text_column", "caption"),
        filename_column=cfg.data.get("filename_column", "filename"),
        batch_size=cfg.data.get("batch_size", 8),
        compute_fad=cfg.evaluation.get("compute_fad", True),
    )

    if loggers:
        for logger in loggers:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(results, step=0)

    log.info(f"Results saved to {data_dir / 'evaluation_results'}")

if __name__ == "__main__":
    main()

