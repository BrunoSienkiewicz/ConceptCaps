import pandas as pd

from datasets import load_dataset
from music_gen_interpretability.data.generic_data_module import GenericDataModule


def transform_concept(df, concept_list, concept_name):
    _df = df.copy()
    for concept in concept_list:
        _df[f"is_{concept_name}_" + concept] = (
            _df["caption"].str.contains(concept, case=False).astype(int)
        )
    return _df


def remove_concept(df, concept_list, concept_name):
    _df = df.copy()
    _df[f"caption_without_{concept_name}"] = _df["caption"]
    for concept in concept_list:
        _df[f"caption_without_{concept_name}"] = _df[
            f"caption_without_{concept_name}"
        ].str.replace(concept, "", case=False)
    _df[f"caption_without_{concept_name}"] = _df[
        f"caption_without_{concept_name}"
    ].str.replace("  ", " ")
    _df[f"caption_without_{concept_name}"] = _df[
        f"caption_without_{concept_name}"
    ].str.strip()
    return _df


class TextConditioning(GenericDataModule):
    def __init__(
        self,
        dataset: str,
        processor: str,
        batch_size: int,
        emotions: list[str],
        instruments: list[str],
        genres: list[str],
    ):
        super().__init__(dataset, processor, batch_size)
        self.emotions = emotions
        self.instruments = instruments
        self.genres = genres

    def prepare_data(self):
        self.dataset = load_dataset(self.dataset)
        self.dataset = self.dataset["train"].to_pandas()
        self.dataset = self.dataset.drop(
            columns=[
                "start_s",
                "end_s",
                "audioset_positive_labels",
                "author_id",
                "is_balanced_subset",
                "is_audioset_eval",
            ]
        )

        self.dataset = transform_concept(self.dataset, self.emotions, "emotion")
        self.dataset = transform_concept(self.dataset, self.instruments, "instrument")
        self.dataset = transform_concept(self.dataset, self.genres, "genre")

        is_any_genre = self.dataset.filter(like="is_genre_").sum(axis=1) > 0
        is_any_instrument = self.dataset.filter(like="is_instrument_").sum(axis=1) > 0
        is_any_emotion = self.dataset.filter(like="is_emotion_").sum(axis=1) > 0

        self.dataset = self.dataset[
            is_any_genre & is_any_instrument & is_any_emotion
        ].reset_index(drop=True)

        self.dataset_genre = remove_concept(self.dataset, self.genres, "genre")
        self.dataset_instrument = remove_concept(
            self.dataset, self.instruments, "instrument"
        )
        self.dataset_emotion = remove_concept(self.dataset, self.emotions, "emotion")
