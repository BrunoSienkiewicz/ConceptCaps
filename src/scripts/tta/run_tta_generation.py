import os
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
from src.tta.audio import generate_audio_samples, generate_audio_samples_accelerate
from src.tta.config import TTAConfig
from src.tta.data import prepare_dataloader, save_dataframe_metadata
from src.tta.modelling import prepare_model, prepare_tokenizer
from src.tta.evaluate import TTAEvaluator


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base=None, config_path="../../../config", config_name="tta_generation")
def main(cfg: TTAConfig):
    # Debug: Check GPU assignment
    if torch.cuda.is_available() and cfg.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Process rank: {local_rank}/{world_size}, Using GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"Total GPUs available: {torch.cuda.device_count()}")
        
        # Set CUDA device based on local rank
        torch.cuda.set_device(local_rank)
        print(f"After set_device - Process {local_rank} using GPU: {torch.cuda.current_device()}")
    
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

    device = torch.device(torch.cuda.current_device() if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    print_config_tree(cfg)

    log.info("Preparing data...")
    processor = prepare_tokenizer(cfg.model)
    dataloader, df = prepare_dataloader(cfg.data, processor)

    log.info("Loading model...")
    model = prepare_model(cfg.model)

    data_dir = Path(cfg.paths.data_dir) / cfg.model.name / cfg.run_id
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating audio samples...")

    if cfg.generation.get("use_accelerator", False):
        log.info("Using Accelerate for distributed generation...")
        generate_audio_samples_accelerate(
            model,
            dataloader,
            data_dir / "audio_samples",
            cfg.model.tokenizer.max_new_tokens,
            cfg.data.batch_size,
            df,
            id_column=cfg.data.get("id_column", "id"),
            filename_template=cfg.data.get("filename_template", "{}.wav"),
            temperature=cfg.generation.get("temperature", 1.0),
            top_k=cfg.generation.get("top_k", 50),
            top_p=cfg.generation.get("top_p", 0.95),
            do_sample=cfg.generation.get("do_sample", True),
            guidance_scale=cfg.generation.get("guidance_scale", None),
        )
    else:
        generate_audio_samples(
            model,
            dataloader,
            data_dir / "audio_samples",
            cfg.model.tokenizer.max_new_tokens,
            cfg.data.batch_size,
            df,
            id_column=cfg.data.get("id_column", "id"),
            filename_template=cfg.data.get("filename_template", "{}.wav"),
            temperature=cfg.generation.get("temperature", 1.0),
            top_k=cfg.generation.get("top_k", 50),
            top_p=cfg.generation.get("top_p", 0.95),
            do_sample=cfg.generation.get("do_sample", True),
            guidance_scale=cfg.generation.get("guidance_scale", None),
        )

    log.info("Saving metadata...")
    save_dataframe_metadata(
        df,
        data_dir,
        id_column=cfg.data.get("id_column", "id"),
        filename_template=cfg.data.get("filename_template", "{}.wav"),
    )

    if not cfg.evaluation.get("skip_evaluation", False):
        return

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

    log.info(f"Results saved to {data_dir / 'evaluation_results'}")

if __name__ == "__main__":
    main()

