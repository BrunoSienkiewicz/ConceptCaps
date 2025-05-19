import torch
from captum.concept import Concept
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from typing import Tuple
from music_gen_interpretability.data.generic_data_module import GenericDataModule

class ConceptDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, concept_tensor: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.concept_tensor = concept_tensor

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.concept_tensor


def create_experimental_set(
    data_module: GenericDataModule,
    num_samples: int,
    experimental_set_size: int,
):
    experimental_set = []
    concept_tensor = torch.full(
        size=(1,),
        fill_value=0,
        dtype=torch.float32,
    )
    data = data_module.select_samples(
        num_samples=num_samples
    )
    concept_dataset = ConceptDataset(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"],
        concept_tensor=concept_tensor,
    )
    concept_dataloader = DataLoader(
        dataset=concept_dataset,
        batch_size=data_module.batch_size,
        shuffle=True,
    )
    influential_concept = Concept(id=0, name=data_module.influential_concept_name, data_iter=concept_dataloader)

    for i in range(1, experimental_set_size + 1):
        random_concept_tensor = torch.full(
            size=(1,),
            fill_value=i,
            dtype=torch.float32,
        )
        data = data_module.select_random_samples(
            num_samples=num_samples
        )
        concept_dataset = ConceptDataset(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            concept_tensor=random_concept_tensor,
        )
        concept_dataloader = DataLoader(
            dataset=concept_dataset,
            batch_size=data_module.batch_size,
            shuffle=True,
        )
        random_concept = Concept(id=i, name=f"random_concept_{i}", data_iter=concept_dataloader)
        experimental_set.append([influential_concept, random_concept])
    return experimental_set
