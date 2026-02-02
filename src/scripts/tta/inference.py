import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import rootutils
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.tta.audio import generate_audio_samples, generate_audio_samples_accelerate
from src.tta.config import TTAConfig
from src.tta.data import prepare_dataloader, save_dataframe_metadata
from src.tta.evaluation import TTAEvaluator
from src.utils import RankedLogger, instantiate_loggers, print_config_tree

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="tta_inference",
)
def main(cfg: TTAConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Debug: Check GPU assignment
    if torch.cuda.is_available() and cfg.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(
            f"Process rank: {local_rank}/{world_size}, Using GPU: {torch.cuda.current_device()}"
        )
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"Total GPUs available: {torch.cuda.device_count()}")

    log.info("Setting random seed...")
    pl.seed_everything(cfg.random_state)

    # Setup loggers
    loggers = []
    if cfg.get("logger"):
        pl_loggers = instantiate_loggers(cfg.logger)
        if pl_loggers:
            loggers.extend(
                pl_loggers if isinstance(pl_loggers, list) else [pl_loggers]
            )

    print_config_tree(cfg)
    log.info("Loading model...")
    processor = AutoProcessor.from_pretrained(cfg.model.name)
    model = MusicgenForConditionalGeneration.from_pretrained(
        cfg.model.name,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    # Compile model for faster inference (PyTorch 2.0+)
    if cfg.generation.get("use_torch_compile", True) and hasattr(
        torch, "compile"
    ):
        log.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    if hasattr(model, "generation_config"):
        model.generation_config.cache_implementation = "static"

    log.info("Preparing data...")
    dataloader, df = prepare_dataloader(cfg.data, processor)

    data_dir = Path(cfg.paths.data_dir) / cfg.model.name / cfg.run_id
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating audio samples...")

    if cfg.generation.get("use_accelerator", True):
        log.info("Using Accelerate for distributed generation...")
        gen_fun = generate_audio_samples_accelerate
    else:
        gen_fun = generate_audio_samples

    if cfg.generation.get("warmup", True):
        log.info("Running warmup generation...")
        with torch.inference_mode():
            dummy_input = processor(
                text=["warmup"], return_tensors="pt", padding=True
            )
            dummy_input = {
                k: v.to(model.device) for k, v in dummy_input.items()
            }
            _ = model.generate(**dummy_input, max_new_tokens=10)
            del dummy_input
            torch.cuda.empty_cache()

    with torch.inference_mode():
        gen_fun(
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
            sample_rate=cfg.generation.get(
                "sample_rate", model.config.audio_encoder.sampling_rate
            ),
            loggers=loggers,
        )

    log.info("Saving metadata...")
    save_dataframe_metadata(
        df,
        data_dir,
        id_column=cfg.data.get("id_column", "id"),
        filename_template=cfg.data.get("filename_template", "{}.wav"),
    )
    log.info(f"Audio samples and metadata saved to {data_dir}")

    if cfg.evaluation.get("skip_evaluation", True):
        return

    log.info("Initializing TTA evaluator...")
    evaluator = TTAEvaluator(
        clap_model=cfg.evaluation.get("clap_model", "laion/clap-htsat-unfused"),
        fad_model=cfg.evaluation.get("fad_model", "laion/clap-htsat-unfused"),
        device=str(device),
    )

    log.info("Running TTA evaluation...")
    results = evaluator.evaluate(
        generated_audio_dir=data_dir / "audio_samples",
        metadata_path=data_dir / "metadata.csv",
        reference_audio_dir=cfg.evaluation.get(
            "reference_audio_dir", data_dir / "reference_audio_samples"
        ),
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
