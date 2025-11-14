from __future__ import annotations

from pathlib import Path

import scipy.io
import torch
import hydra
import pytorch_lightning as pl

from src.tta.config import TTAConfig
from src.data.tta_dataset import load_and_tokenize_dataset, get_dataloader
from src.utils import RankedLogger, instantiate_loggers


log = RankedLogger(__name__, rank_zero_only=True)


def run_tta(cfg: TTAConfig) -> None:
    """Generate music from text descriptions.
    
    Args:
        cfg: TTA configuration with model, data, device, and output paths.
    """
    pl.seed_everything(cfg.random_state)

    log.info("Instantiating loggers...")
    _ = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    # Load and tokenize dataset
    log.info("Loading and tokenizing dataset...")
    processor = hydra.utils.instantiate(cfg.data.processor)
    dataset, metadata_df = load_and_tokenize_dataset(
        dataset_name=cfg.data.dataset,
        processor=processor,
        subset=cfg.data.get("subset", "train"),
        subset_size=cfg.data.get("subset_size", 0.1),
        max_sequence_length=cfg.data.get("max_sequence_length", 256),
        caption_column=cfg.data.get("caption_column", "caption"),
        device=device,
    )

    # Load model
    log.info("Loading model...")
    model = hydra.utils.instantiate(cfg.model.model)
    model.to(device)

    output_root = Path(cfg.paths.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save dataset metadata
    metadata_df.to_csv(output_root / "full_dataset.csv", index=False)
    log.info(f"Saved dataset metadata: {len(metadata_df)} samples")

    # Generate audio
    log.info("Generating audio samples...")
    output_dir = output_root / "tta_generation"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(dataset, batch_size=cfg.data.batch_size, shuffle=False)
    max_new_tokens = cfg.model.model.max_new_tokens

    for batch_idx, batch in enumerate(dataloader):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            audio_values = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

        sampling_rate = model.model.config.audio_encoder.sampling_rate
        for item_idx, audio in enumerate(audio_values):
            global_idx = batch_idx * cfg.data.batch_size + item_idx
            scipy.io.wavfile.write(
                output_dir / f"{global_idx}.wav",
                sampling_rate,
                audio[0].cpu().numpy(),
            )

    log.info(f"TTA generation completed. Generated {len(dataset)} samples in {output_dir}")
