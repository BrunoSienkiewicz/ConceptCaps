import torch
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import AutoProcessor
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
        batch_size: int,
        processor: AutoProcessor,
        emotions: list[str],
        instruments: list[str],
        genres: list[str],
        influential_concept_name: str,
        influential_concept_category: str,
        target_concept_name: str,
        target_concept_category: str,
    ):
        self.processor = processor
        self.emotions = emotions
        self.instruments = instruments
        self.genres = genres
        self.influential_concept_name = influential_concept_name
        self.influential_concept_category = influential_concept_category
        self.target_concept_name = target_concept_name
        self.target_concept_category = target_concept_category
        super().__init__(dataset, batch_size)

    def _transform(self, dataset: pd.DataFrame):
        dataset = dataset.drop(
            columns=[
                "start_s",
                "end_s",
                "audioset_positive_labels",
                "author_id",
                "is_balanced_subset",
                "is_audioset_eval",
            ]
        )

        dataset = transform_concept(dataset, self.emotions, "emotion")
        dataset = transform_concept(dataset, self.instruments, "instrument")
        dataset = transform_concept(dataset, self.genres, "genre")

        is_any_genre = dataset.filter(like="is_genre_").sum(axis=1) > 0
        is_any_instrument = dataset.filter(like="is_instrument_").sum(axis=1) > 0
        is_any_emotion = dataset.filter(like="is_emotion_").sum(axis=1) > 0

        dataset = dataset[
            is_any_genre & is_any_instrument & is_any_emotion
        ].reset_index(drop=True)

        if self.target_concept_category == "emotion":
            target_category_words = self.emotions
        elif self.target_concept_category == "instrument":
            target_category_words = self.instruments
        elif self.target_concept_category == "genre":
            target_category_words = self.genres
        dataset = remove_concept(dataset, target_category_words, self.target_category)
        return dataset

    def _tokenize(self, text: list[str]) -> dict[str, torch.Tensor]:
        inputs = self.processor(
            text=text,
            max_length=256,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        # {"input_ids": ..., "attention_mask": ...}
        return inputs

    def prepare_data(self):
        self.dataset_loaded = load_dataset(self.dataset)
        self.dataset_transformed = self._transform(self.dataset_loaded["train"].to_pandas())

    def setup(self):
        self.dataset_concept = self.dataset_transformed[
            (self.dataset_transformed[f"is_{self.influential_concept_category}_" + self.influential_concept_name] == 1 &
                self.dataset_transformed[f"is_{self.target_concept_category}_" + self.target_concept_name] == 1)
        ].reset_index(drop=True)
        self.dataset_concept = self._tokenize(self.dataset_concept[f"caption_without_{self.target_concept_category}"].tolist())
        self.dataset_all = self.dataset_transformed
        self.dataset_all = self._tokenize(self.dataset_all[f"caption_without_{self.target_concept_category}"].tolist())   

    def select_samples(self, num_samples: int):
        if num_samples > len(self.dataset_concept):
            raise ValueError("Number of samples requested exceeds the dataset size.")
        indices = np.random.choice(len(self.dataset_concept), num_samples, replace=False)
        selected_samples = {
            "input_ids": self.dataset_concept["input_ids"][indices],
            "attention_mask": self.dataset_concept["attention_mask"][indices],
        }
        return selected_samples

    def select_random_samples(self, num_samples: int):
        if num_samples > len(self.dataset_all):
            raise ValueError("Number of samples requested exceeds the dataset size.")
        indices = np.random.choice(len(self.dataset_all), num_samples, replace=False)
        random_samples = {
            "input_ids": self.dataset_all["input_ids"][indices],
            "attention_mask": self.dataset_all["attention_mask"][indices],
        }
        return random_samples
