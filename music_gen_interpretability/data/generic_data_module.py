from transformers import AutoProcessor

class GenericDataModule:
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        processor: AutoProcessor = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.processor = processor

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        # This method is called to download the data if needed
        pass

    def setup(self, stage=None):
        # This method is called to set up the data for training, validation, and testing
        pass

    def select_samples(self, num_samples: int):
        # This method should return a subset of the dataset with the specified number of samples
        pass

    def select_random_samples(self, num_samples: int):
        # This method should return a random subset of the dataset with the specified number of samples
        pass
