import pandas as pd


def select_samples(
    df: pd.DataFrame,
    concept: str,
    genre: str,
    num_samples: int,
    random_state: int = 42,
):
    _df = df.copy()
    _df = _df[_df[f"is_genre_{genre}"] == 1]
    _df = _df[_df[f"is_{concept}"] == 1]
    _df = _df.sample(num_samples, random_state=random_state)
    return _df


def select_random_samples(
    df: pd.DataFrame,
    num_samples: int,
    random_state: int = 42,
):
    _df = df.copy()
    _df = _df.sample(num_samples, random_state=random_state)
    return _df


#####################################
# Text-conditioned music generation #
#####################################


def transform_concept(
    df: pd.DataFrame, concept_list: list[str], concept_name: str
) -> pd.DataFrame:
    _df = df.copy()
    for concept in concept_list:
        _df[f"is_{concept_name}_" + concept] = (
            _df["caption"].str.contains(concept, case=False).astype(int)
        )
    return _df


def remove_concept(df: pd.DataFrame, concept_list: list[str], concept_name: str) -> pd.DataFrame:
    _df = df.copy()
    _df[f"caption_without_{concept_name}"] = _df["caption"]
    for concept in concept_list:
        _df[f"caption_without_{concept_name}"] = _df[
            f"caption_without_{concept_name}"
        ].str.replace(concept, "", case=False)
    _df[f"caption_without_{concept_name}"] = _df[f"caption_without_{concept_name}"].str.replace(
        "  ", " "
    )
    _df[f"caption_without_{concept_name}"] = _df[f"caption_without_{concept_name}"].str.strip()
    return _df
