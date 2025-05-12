from dataclasses import dataclass


@dataclass
class GenericDataModule:
    data_path: str
    batch_size: int
    num_samples: int
    experimental_set_size: int
