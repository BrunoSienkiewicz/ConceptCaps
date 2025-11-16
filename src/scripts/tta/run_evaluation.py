from __future__ import annotations

import hydra
import rootutils
import torch
from pathlib import Path

from src.tta.config import TTAConfig
from src.tta.evaluate import TTAEvaluator
from src.utils import print_config_tree, RankedLogger, instantiate_loggers


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tta")
def main(cfg: TTAConfig) -> None:
    """Main evaluation function for TTA generation."""
    
    # Set random seed for reproducibility
    import pytorch_lightning as pl
    pl.seed_everything(cfg.random_state)

    # Setup device
    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Print configuration
    print_config_tree(cfg)

    # Setup loggers
    if cfg.get("logger"):
        loggers = instantiate_loggers(cfg.logger)

    # Setup paths
    generated_audio_dir = Path(cfg.paths.data_dir) / "audio_samples"
    metadata_path = Path(cfg.paths.data_dir) / "metadata.csv"
    output_dir = Path(cfg.paths.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate paths
    if not generated_audio_dir.exists():
        raise FileNotFoundError(
            f"Generated audio directory not found: {generated_audio_dir}. "
            "Please run audio generation first."
        )
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}. "
            "Please run audio generation first."
        )

    # Initialize evaluator
    log.info("Initializing TTA evaluator...")
    evaluator = TTAEvaluator(
        clap_model=cfg.evaluation.get("clap_model", "laion/clap-htsat-unfused"),
        fad_model=cfg.evaluation.get("fad_model", "google/vggish"),
        device=str(device),
    )

    # Get reference audio directory or background stats path
    reference_audio_dir = None
    background_stats_path = None
    
    if cfg.evaluation.get("reference_audio_dir"):
        reference_audio_dir = Path(cfg.evaluation.reference_audio_dir)
        if not reference_audio_dir.exists():
            log.warning(f"Reference audio directory not found: {reference_audio_dir}")
            reference_audio_dir = None
    
    if cfg.evaluation.get("background_stats_path"):
        background_stats_path = Path(cfg.evaluation.background_stats_path)
        if not background_stats_path.exists():
            log.warning(f"Background stats file not found: {background_stats_path}")
            background_stats_path = None

    # Run evaluation
    log.info("Running TTA evaluation...")
    results = evaluator.evaluate(
        generated_audio_dir=generated_audio_dir,
        metadata_path=metadata_path,
        reference_audio_dir=reference_audio_dir,
        background_stats_path=background_stats_path,
        output_dir=output_dir,
        text_column=cfg.data.get("caption_column", "caption"),
        filename_column=cfg.data.get("filename_column", "filename"),
        batch_size=cfg.evaluation.get("batch_size", 8),
    )

    # Log results
    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info("=" * 50)
    for metric_name, metric_value in results.items():
        log.info(f"{metric_name}: {metric_value:.4f}")
    log.info("=" * 50)

    log.info(f"Evaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
