from pathlib import Path

import wandb
import time
import hydra
import rootutils
import torch
import pytorch_lightning as pl

from src.utils import (RankedLogger, instantiate_loggers,
                        instantiate_callbacks,
                       print_config_tree)
from src.tta.audio import generate_audio_samples
from src.tta.config import TTAConfig
from src.tta.data import prepare_dataloader, save_dataframe_metadata
from src.tta.modelling import prepare_model, prepare_tokenizer
from src.tta.evaluate import TTAEvaluator


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tta_generation")
def main(cfg: TTAConfig):
    log.info("Setting random seed...")
    pl.seed_everything(cfg.random_state)

    # Setup callbacks
    callbacks = []
    if cfg.get("callbacks"):
        pl_callbacks = instantiate_callbacks(cfg.callbacks)
        if pl_callbacks:
            callbacks.extend(pl_callbacks if isinstance(pl_callbacks, list) else [pl_callbacks])

    # Setup loggers
    loggers = []
    if cfg.get("logger"):
        pl_loggers = instantiate_loggers(cfg.logger)
        if pl_loggers:
            loggers.extend(pl_loggers if isinstance(pl_loggers, list) else [pl_loggers])

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    log.info("Preparing data...")
    processor = prepare_tokenizer(cfg.model)
    dataloader, df = prepare_dataloader(cfg.data, processor)

    log.info("Loading model...")
    model = prepare_model(cfg.model)
    model.to(device)

    data_dir = Path(cfg.paths.data_dir) / cfg.model.name / cfg.run_id
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating audio samples...")
    generate_audio_samples(
        model,
        dataloader,
        data_dir / "audio_samples",
        cfg.model.model.max_new_tokens,
        cfg.data.batch_size,
        df,
        id_column=cfg.data.get("id_column", "id"),
        filename_template=cfg.data.get("filename_template", "{}.wav"),
    )

    log.info("Saving metadata...")
    save_dataframe_metadata(
        df,
        data_dir,
        id_column=cfg.data.get("id_column", "id"),
        filename_template=cfg.data.get("filename_template", "{}.wav"),
    )

    # Initialize evaluator
    log.info("Initializing TTA evaluator...")
    evaluator = TTAEvaluator(
        clap_model=cfg.evaluation.get("clap_model", "laion/clap-htsat-unfused"),
        fad_model=cfg.evaluation.get("fad_model", "google/vggish"),
        device=str(device),
    )

    log.info("TTA generation completed and logged.")

    log.info("Running TTA evaluation...")
    results = evaluator.evaluate(
        generated_audio_dir=data_dir / "audio_samples",
        metadata_path=data_dir / "metadata.csv",
        output_dir=data_dir / "evaluation_results",
        text_column=cfg.data.get("caption_column", "caption"),
        filename_column=cfg.data.get("filename_column", "filename"),
        batch_size=cfg.data.get("batch_size", 8),
    )

    if loggers:
        for logger in loggers:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(results, step=0)

    # Log results
    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info("=" * 50)
    for metric_name, metric_value in results.items():
        log.info(f"{metric_name}: {metric_value:.4f}")
    log.info("=" * 50)

    log.info(f"Evaluation completed! Results saved to {data_dir / 'evaluation_results'}")

if __name__ == "__main__":
    main()

