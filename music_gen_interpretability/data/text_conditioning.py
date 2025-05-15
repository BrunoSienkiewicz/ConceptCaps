import pandas as pd

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
    ):
        super().__init__(dataset, processor, batch_size)
        self.emotions = emotions
        self.instruments = instruments
        self.genres = genres

    def _tranform(self, dataset: pd.DataFrame):
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

        dataset = remove_concept(dataset, self.genres, "genre")
        dataset = remove_concept(
            dataset, self.instruments, "instrument"
        )
        dataset = remove_concept(dataset, self.emotions, "emotion")
        return dataset

    def _tokenize(self, text: list[str]):
        inputs = self.processor(
            text=text,
            max_length=256,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_ids, attention_mask

    def prepare_data(self):
        self.dataset = load_dataset(self.dataset)

        self.dataset_train = self._tranform(self.dataset["train"])
        self.dataset_test = self._tranform(self.dataset["test"])
        self.dataset_valid = self._tranform(self.dataset["validation"])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = {
                "genre": self._tokenize(self.dataset_train["caption_without_genre"].tolist()),
                "emotion": self._tokenize(self.dataset_train["caption_without_emotion"].tolist()),
                "instrument": self._tokenize(self.dataset_train["caption_without_instrument"].tolist()),
            }
            self.val_dataset = {
                "genre": self._tokenize(self.dataset_train["caption_without_genre"].tolist()),
                "emotion": self._tokenize(self.dataset_train["caption_without_emotion"].tolist()),
                "instrument": self._tokenize(self.dataset_train["caption_without_instrument"].tolist()),
            } 
        if stage == "test" or stage is None:
            self.train_dataset = {
                "genre": self._tokenize(self.dataset_train["caption_without_genre"].tolist()),
                "emotion": self._tokenize(self.dataset_train["caption_without_emotion"].tolist()),
                "instrument": self._tokenize(self.dataset_train["caption_without_instrument"].tolist()),
            }
