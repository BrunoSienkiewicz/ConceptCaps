import lightning as L


class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        processor: str,
        batch_size: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size
