import pytorch_lightning as pl


class GenericDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        batch_size: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def select_samples(self, num_samples: int):
        # This method should return a subset of the dataset with the specified number of samples
        pass

    def select_random_samples(self, num_samples: int):
        # This method should return a random subset of the dataset with the specified number of samples
        pass
