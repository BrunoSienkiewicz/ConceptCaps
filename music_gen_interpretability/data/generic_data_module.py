import pandas as pd
import lightning as L


class GenericDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, num_samples: int, experimental_set_size: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = num_samples
