import numpy as np
import torch

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from captum.concept._utils.classifier import Classifier
from torch.utils.data import DataLoader
from transformers import MusicgenModel, MusicgenProcessor


class ConceptClassifier(Classifier):
    def __init__(self, random_state: int = 42, *args, **kwargs):
        self.lm = SGDClassifier(
            *args,
            **kwargs,
            random_state=random_state,
        )

    def train_and_eval(self, dataloader: DataLoader, test_split_ratio = 0.33, random_state: int = 42, **kwargs) -> dict:
        X = []
        y = []
        for batch in dataloader:
            x, y_batch = batch
            X.append(x.cpu().numpy())
            y.append(y_batch.cpu().numpy())

        # Pad the sequences to the same length
        max_len = max([x.shape[1] for x in X])
        X = [np.pad(x, ((0, 0), (0, max_len - x.shape[1])), mode='constant') for x in X]

        X = np.concatenate(X)
        y = np.concatenate(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_split_ratio,
            random_state=random_state,
        )

        self.lm.fit(X_train, y_train)
        y_pred = self.lm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return {"accuracy": acc}

    def weights(self) -> torch.Tensor:
        if len(self.lm.coef_) == 1:
            # if there are two concepts, there is only one label.
            # We split it in two.
            return torch.tensor([-1 * self.lm.coef_[0], self.lm.coef_[0]])
        else:
            return torch.tensor(self.lm.coef_)

    def classes(self) -> list[int]:
        return list(self.lm.classes_)


class CustomMusicGen:
    def __init__(self, model: MusicgenModel, processor: MusicgenProcessor, max_new_tokens=256, device: str = "cpu"):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

        self.model.eval()
        self.model.to(device)

    def __call__(self, input_ids, attention_mask, concept_tensor):
        return self.forward(input_ids, attention_mask, concept_tensor)

    def forward(self, input_ids, attention_mask, concept_tensor):
        torch.cuda.empty_cache()
        with torch.amp.autocast('cuda'):
            audio_values = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                # output_hidden_states=True,
                # return_dict_in_generate=True,
                # use_cache=False,
            )
        return audio_values[0]
