import pandas as pd
import torch
from captum.concept import Concept
from torch.utils.data import DataLoader

from music_gen_interpretability.tcav.dataset import ConceptDataset
from music_gen_interpretability.tcav.transform import (
    select_random_samples,
    select_samples,
)


def create_experimental_set(
    concept_name,
    genre,
    data_module,
    experimental_set_size,
):
    experimental_set = []
    concept_tensor = torch.full(
        size=(1,),
        fill_value=0,
        dtype=torch.float32,
    )
    concept = data_module.assemble_concept(
        name=concept_name,
        id=0,
        concept_name=concept_name,
        genre=genre,
        concept_tensor=concept_tensor,
    )

    for i in range(1, experimental_set_size + 1):
        random_concept_tensor = torch.full(
            size=(1,),
            fill_value=i,
            dtype=torch.float32,
        )
        random_concept = data_module.assemble_random_concept(
            name=f"random_{i}",
            id=i,
            concept_tensor=random_concept_tensor,
        )
        experimental_set.append([concept, random_concept])
    return experimental_set
