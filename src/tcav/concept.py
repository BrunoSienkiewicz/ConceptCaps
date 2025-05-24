import torch
from captum.concept import Concept

from src.data.generic_data_module import GenericDataModule


def create_experimental_set(
    data_module: GenericDataModule,
    num_samples: int,
    experimental_set_size: int,
    device: torch.device = torch.device("cpu"),
):
    experimental_set = []
    concept_tensor = torch.full(
        size=(1,),
        fill_value=0,
        dtype=torch.float32,
    )
    concept_dataloader = data_module.concept_dataloader(
        num_samples=num_samples,
        concept_tensor=concept_tensor,
    )
    influential_concept = Concept(
        id=0, name=data_module.influential_concept_name, data_iter=concept_dataloader
    )

    for i in range(1, experimental_set_size + 1):
        random_concept_tensor = torch.full(
            size=(1,),
            fill_value=i,
            dtype=torch.float32,
        )
        concept_dataloader = data_module.random_dataloader(
            num_samples=num_samples,
            concept_tensor=random_concept_tensor,
        )
        random_concept = Concept(
            id=i, name=f"random_concept_{i}", data_iter=concept_dataloader
        )
        experimental_set.append([influential_concept, random_concept])
    return experimental_set
