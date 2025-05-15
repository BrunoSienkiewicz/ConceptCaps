import lightning as L


class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        batch_size: int,
        processor: AuroProcessor = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.processor = processor

    def select_samples(
        self,
        influential_concept_name: str,
        influential_concept_category: str,
        target_concept_name: str,
        target_concept_category: str,
        num_samples: int,
    ):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def select_random_samples(
        self,
        influential_concept_name: str,
        influential_concept_category: str,
        target_concept_name: str,
        target_concept_category: str,
        num_samples: int,
    ):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_experimental_set(
        self,
        influential_concept_name: str,
        influential_concept_category: str,
        target_concept_name: str,
        target_concept_category: str,
        num_samples: int,
    ):
        raise NotImplementedError("This method should be implemented by subclasses.")
