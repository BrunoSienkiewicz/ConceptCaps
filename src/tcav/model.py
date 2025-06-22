import copy
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from captum.concept._utils.classifier import Classifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanMetric
from transformers import MusicgenModel, MusicgenProcessor
from transformers.generation.configuration_utils import (GenerationConfig,
                                                         GenerationMode)
from transformers.generation.logits_process import (
    ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

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
        X = [
            np.pad(x, ((0, 0), (0, max_len - x.shape[1])), mode="constant")
            for x in X
        ]

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
    def __init__(self, input_size: int, cav_size: int, num_classes: int, learning_rate: float = 0.001):
        super().__init__()

        self.save_hyperparameters()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, cav_size),
            torch.nn.ReLU(),
            # I add +1 to the number of classes to account for the researched class
            # num_classes is derived from experimental_set_size
            torch.nn.Linear(cav_size, num_classes + 1),
        )
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes + 1 
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes + 1
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes + 1
        )

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_accuracy.reset()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        if len(y.shape) == 2:
            y = y.squeeze(-1)
        logits = logits.squeeze(-1)
        y_pred = torch.softmax(logits, dim=1).float()
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self.model_step(batch)
        self.train_accuracy(y_pred, y)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self.model_step(batch)
        self.val_accuracy(y_pred, y)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self.model_step(batch)
        self.test_accuracy(y_pred, y)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class NetClassifier(Classifier):
    def __init__(
        self, 
        input_size: int, # must match the input/output of resarched layer
        cav_size: int, # size of activation vector
        num_classes: int, # must match the number of concepts
        trainer: pl.Trainer,
        batch_size: int = 8,
        test_split_ratio=0.33,
        val_split_ratio=0.2,
        random_state: int = 42,
        *args, 
        **kwargs
    ):
        self.model = CustomNet(
            input_size=input_size,
            cav_size=cav_size,
            num_classes=num_classes,
        )
        self.batch_size = batch_size
        self.trainer = trainer
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio = val_split_ratio
        self.random_state = random_state
        self.train_and_eval_calls = 0

        super().__init__()

    def train_and_eval(
        self,
        dataloader: DataLoader,
        *args,
        **kwargs,
    ) -> dict:
        self.train_and_eval_calls += 1
        log.info(f"Training and evaluating classifier, call #{self.train_and_eval_calls}") 
        X = []
        y = []

        for batch in dataloader:
            x, y_batch = batch
            X.append(x)
            y.append(y_batch)
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        dataset = torch.utils.data.TensorDataset(X, y)

        test_amount, val_amount = int(dataset.__len__() * self.test_split_ratio), int(dataset.__len__() * self.val_split_ratio)

        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
                    (dataset.__len__() - (test_amount + val_amount)), 
                    test_amount, 
                    val_amount
        ])
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.trainer.fit(self.model, train_loader, val_loader)
        test_result = self.trainer.test(self.model, test_loader)
        print(test_result)
        acc = test_result[0]["test/accuracy"]
        return {"accuracy": acc}

    def weights(self) -> torch.Tensor:
        return self.model.net[0].weight.data.squeeze(0)

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


class MusicGenWithGrad(pl.LightningModule):
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

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        audio_values = self.generate_with_grad(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
        )
        return audio_values[0]

    def generate_with_grad(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.model.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(model_kwargs.copy())

        if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) is tuple:
            # wrap the unconditional outputs as a BaseModelOutput for compatibility with the rest of generate
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        self.model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

        # 4. Define other model kwargs
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )

        if "encoder_outputs" not in model_kwargs:
            # encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self.model._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs:
            model_kwargs = self.model._prepare_audio_encoder_kwargs_for_generation(
                model_kwargs["input_values"],
                model_kwargs,
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self.model._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config._decoder_start_token_tensor,
            bos_token_id=generation_config._bos_token_tensor,
            device=inputs_tensor.device,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self.model._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to MusicGen)
        input_ids, decoder_delay_pattern_mask = self.model.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=generation_config._decoder_start_token_tensor,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        # input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode()

        # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None

        # 9. prepare distribution pre_processing samplers
        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
            device=input_ids.device,
        )

        # 10. prepare stopping criteria
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 11. run sample
            outputs = self.model._sample(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # apply the pattern mask to the final ids
        output_ids = self.model.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # revert the pattern delay mask by filtering the pad token id
        output_ids = output_ids[output_ids != generation_config._pad_token_tensor].reshape(
            batch_size, self.model.decoder.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...]

        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size

        if self.model.decoder.config.audio_channels == 1:
            output_values = self.model.audio_encoder.decode(
                output_ids,
                audio_scales=audio_scales,
            ).audio_values
        else:
            codec_outputs_left = self.model.audio_encoder.decode(output_ids[:, :, ::2, :], audio_scales=audio_scales)
            output_values_left = codec_outputs_left.audio_values

            codec_outputs_right = self.model.audio_encoder.decode(output_ids[:, :, 1::2, :], audio_scales=audio_scales)
            output_values_right = codec_outputs_right.audio_values

            output_values = torch.cat([output_values_left, output_values_right], dim=1)

        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values
