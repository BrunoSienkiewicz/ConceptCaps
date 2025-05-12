from dataclasses import dataclass
from music_gen_interpretability.data.generic_data_module import GenericDataModule


@dataclass
class TextConditioning(GenericDataModule):
    data_path: str
    batch_size: int
    num_samples: int
    experimental_set_size: int
