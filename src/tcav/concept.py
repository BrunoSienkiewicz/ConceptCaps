import pandas as pd
import torch

from torch.utils.data import DataLoader
from captum.concept import Concept

from .dataset import ConceptDataset
from .transform import select_samples, select_random_samples


def assemble_concept(name, id, concept_name, genre, data_path, batch_size, concept_tensor, num_samples=None):
    df = pd.read_csv(data_path)
    concept_df = select_samples(
        df=df,
        concept=concept_name,
        genre=genre,
        num_samples=num_samples,
    )
    concept_dataset = ConceptDataset(
        caption_column="caption_without_genre",
        df=concept_df,
        concept_tensor=concept_tensor,
    )
    concept_dataloader = DataLoader(
        concept_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return Concept(id=id, name=name, data_iter=concept_dataloader)

def assemble_random_concept(name, id, data_path, batch_size, concept_tensor, num_samples=None):
    df = pd.read_csv(data_path)
    concept_df = select_random_samples(
        df=df,
        num_samples=num_samples,
    )
    concept_dataset = ConceptDataset(
        caption_column="caption_without_genre",
        df=concept_df,
        concept_tensor=concept_tensor,
    )
    concept_dataloader = DataLoader(
        concept_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return Concept(id=id, name=name, data_iter=concept_dataloader)

def create_experimental_set(
    concept_name,
    genre,
    data_path,
    batch_size,
    num_samples,
    experimental_set_size,
):
    experimental_set = []
    concept_tensor = torch.full(
        size=(1,),
        fill_value=0,
        dtype=torch.float32,
    )
    concept = assemble_concept(
        name=concept_name,
        id=0,
        concept_name=concept_name,
        genre=genre,
        data_path=data_path,
        batch_size=batch_size,
        concept_tensor=concept_tensor,
        num_samples=num_samples,
    )

    for i in range(1, experimental_set_size + 1):
        random_concept_tensor = torch.full(
            size=(1,),
            fill_value=i,
            dtype=torch.float32,
        )
        random_concept = assemble_random_concept(
            name=f"random_{i}",
            id=i,
            data_path=data_path,
            batch_size=batch_size,
            concept_tensor=random_concept_tensor,
            num_samples=num_samples,
        )
        experimental_set.append([concept, random_concept])
    return experimental_set
