from typing import Optional
import numpy as np
import torch
import pytorch_lightning as pl
from captum.concept._utils.classifier import Classifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import MusicgenModel, MusicgenProcessor
from torchmetrics import Accuracy


class SVMClassifier(Classifier):
    def __init__(self, random_state: int = 42, *args, **kwargs):
        super().__init__()

        self.lm = SGDClassifier(
            random_state=random_state,
        )

    def train_and_eval(
        self,
        dataloader: DataLoader,
        test_split_ratio=0.33,
        random_state: int = 42,
        *args,
        **kwargs,
    ) -> dict:
        X = []
        y = []
        for batch in dataloader:
            x, y_batch = batch
            X.append(x.cpu().numpy())
            y.append(y_batch.cpu().numpy())

        # Pad the sequences to the same length
        max_len = max([x.shape[1] for x in X])
        X = [np.pad(x, ((0, 0), (0, max_len - x.shape[1])), mode="constant") for x in X]

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


class CustomNet(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int, num_classes: int):
        super().__init__()

        self.save_hyperparameters()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        )
        self.num_classes = num_classes
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        print(logits.min(), logits.max(), logits.shape)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        self.train_accuracy(logits, y)
        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        print(logits.min(), logits.max(), logits.shape)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        acc = self.val_accuracy(logits, y)
        self.log("val_accuracy", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        print(logits.min(), logits.max(), logits.shape)
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(logits, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



class NetClassifier(Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def train_and_eval(
        self,
        dataloader: DataLoader,
        trainer: Optional[pl.Trainer] = None,
        *args, **kwargs
    ) -> dict:
        if trainer is None:
            # If no trainer is provided, create a default one
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = pl.Trainer(
                max_epochs=10,
                logger=False,
                enable_progress_bar=False,
                accelerator=self.device,
                devices=1 if self.device == "cuda" else None,
            )

        # Values are hardcoded for now.
        # Later it should be changed based on experiment config.
        self.model = CustomNet(
            input_dim=2048,
            output_dim=4,
            num_classes=4,
        )

        trainer.fit(self.model, dataloader)
        test_result = trainer.test(self.model, dataloader)
        acc = test_result[0]["test_accuracy"]
        return {"accuracy": acc}

    def weights(self) -> torch.Tensor:
        weights = self.model.net[-1].weight.data.cpu().numpy().flatten().tolist()
        if len(weights) == 1:
            # if there are two concepts, there is only one label.
            # We split it in two.
            return torch.tensor([-1 * weights[0], weights[0]])
        else:
            return torch.tensor(weights)

    def classes(self) -> list[int]:
        return list(range(self.model.num_classes))


class CustomMusicGen(pl.LightningModule):
    def __init__(
        self,
        model: MusicgenModel,
        processor: MusicgenProcessor,
        max_new_tokens=256,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_ids, attention_mask, *args, **kwargs):
        return self.forward(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        audio_values = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
        )
        return audio_values[0]
