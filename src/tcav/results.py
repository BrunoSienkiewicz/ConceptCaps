from __future__ import annotations

from typing import Dict, List

import pandas as pd
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str


def format_float(value: float) -> float:
    return float(f"{value:.3f}" if abs(value) >= 0.0005 else f"{value:.3e}")


def build_results_dataframe(
    experimental_sets: List[List[Concept]],
    tcav_scores: Dict[str, Dict[str, Dict[str, List[float]]]],
) -> pd.DataFrame:
    scores: List[float] = []
    concepts: List[str] = []
    layers: List[str] = []
    sets: List[int] = []

    for set_idx, concept_group in enumerate(experimental_sets):
        concepts_key = concepts_to_str(concept_group)
        for concept_idx, concept in enumerate(concept_group):
            for layer, layer_scores in tcav_scores[concepts_key].items():
                value = format_float(layer_scores["sign_count"][concept_idx])
                scores.append(value)
                concepts.append(concept.name)
                layers.append(layer)
                sets.append(set_idx)

    dataframe = pd.DataFrame.from_dict(
        {
            "Scores": scores,
            "Concept": concepts,
            "Layer": layers,
            "Set": sets,
        }
    )

    dataframe["Scores"] = dataframe["Scores"].astype(float)
    dataframe["Set"] = dataframe["Set"].astype(int)
    dataframe["Layer"] = dataframe["Layer"].astype(str)
    dataframe["Concept"] = dataframe["Concept"].astype(str)
    return dataframe
