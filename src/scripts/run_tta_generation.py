from pathlib import Path

import wandb
import hydra
import scipy.io
import rootutils
import torch
import scipy
import pytorch_lightning as pl

from src.utils import (RankedLogger, instantiate_loggers, log_hyperparameters,
                       print_config_tree)
from src.tta.config import TTAConfig
from src.tta.helper_methods import preprocess_dataset, sample_dataset, generate_audio

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)

def tta(cfg: TTAConfig):
    random_state = cfg.random_state
    pl.seed_everything(random_state)

    log.info("Instantiating loggers...")
    wandb.login()
    logger = instantiate_loggers(cfg.get("logger"))

    device = torch.device(cfg.device)
    log.info(f"Using device: {device}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.model.to(device)

    object_dict = {
        "cfg": cfg,
        "data_module": data_module,
        "model": model,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Preparing data...")
    data_module.prepare_data()
    data_module.setup()

    data_module.dataset.to_csv(Path(cfg.paths.output_dir) / "full_dataset.csv", index=False)

    for i, batch in enumerate(data_module.random_dataloader()):
        input_ids, attention_mask = batch
        audio_values = model.model(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=cfg.model.model.max_new_tokens)
        sampling_rate = model.model.model.config.audio_encoder.sampling_rate
        output_dir = Path(cfg.paths.output_dir) / "tta_generation"
        output_dir.mkdir(parents=True, exist_ok=True)
        for j, audio in enumerate(audio_values):
            idx = i * data_module.batch_size + j
            scipy.io.wavfile.write(output_dir / f"{idx}.wav", sampling_rate, audio[0].cpu().numpy())


@hydra.main(version_base=None, config_path="../../config", config_name="tta")
def main(cfg: TTAConfig):
    print_config_tree(cfg)
    tta(cfg)


if __name__ == "__main__":
    main()
