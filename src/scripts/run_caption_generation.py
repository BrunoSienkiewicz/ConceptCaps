from __future__ import annotations

import hydra
import rootutils

from src.caption import CaptionGenerationConfig, run_caption_generation
from src.utils import print_config_tree


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base=None, config_path="../../config", config_name="caption_generation")
def main(cfg: CaptionGenerationConfig) -> None:
    print_config_tree(cfg)
    run_caption_generation(cfg)


if __name__ == "__main__":
    main()
