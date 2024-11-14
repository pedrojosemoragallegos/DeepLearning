from typing import Optional
import torch
from torch import device as Device
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module as Model
from torch.nn.modules.loss import _Loss as Criterion
from .callback import CallbackList
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler


def log_callback(callbacks: CallbackList, method: str, **kwargs):
    getattr(callbacks, method)(**kwargs)


class Trainer:
    def __init__(
        self,
        model: Model,
        device: Device,
        criterion: Criterion,
        train_dataloader: Optional[DataLoader] = None,
        validation_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scaler: Optional[GradScaler] = None,
    ):
        self._model: Model = model
        self._device: Device = device
        self._criterion: Criterion = criterion
        self._train_dataloader: Optional[DataLoader] = train_dataloader
        self._validation_dataloader: Optional[DataLoader] = validation_dataloader
        self._test_dataloader: Optional[DataLoader] = test_dataloader
        self._optimizer: Optional[Optimizer] = optimizer
        self._scaler: Optional[GradScaler] = scaler

        self._callbacks: Optional[CallbackList] = None
        self._epoch_num: Optional[int] = None
        self._batch_index: Optional[int] = None
        self._train_batch_loss: Optional[float] = None
        self._validation_batch_loss: Optional[float] = None
        self._test_batch_loss: Optional[float] = None
        self._train_total_batch_loss: Optional[float] = None
        self._validation_total_batch_loss: Optional[float] = None
        self._train_batch_index: Optional[int] = None
        self._test_batch_index: Optional[int] = None
        self._validation_batch_index: Optional[int] = None
        self._train_epoch_loss: Optional[float] = None

        self._train_epoch_num: Optional[int] = None

    def _process_batch(
        self, inputs: Tensor, labels: Tensor, optimizer: Optional[Optimizer] = None
    ):
        if optimizer:
            optimizer.zero_grad()

        if self._callbacks:
            log_callback(  # TODO pass evertything
                self._callbacks,
                "on_train_batch_start" if optimizer else "on_validation_batch_start",
            )

        with autocast(enabled=self._scaler is not None):
            outputs: Tensor = self._model(inputs)
            loss: Tensor = self._criterion(outputs, labels)

        if optimizer:
            if self._scaler:
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                loss.backward()
                optimizer.step()

            self._train_batch_loss = loss.item()
        else:
            self._validation_batch_loss = loss.item()

        if self._callbacks:
            log_callback(  # TODO pass everything
                self._callbacks,
                "on_train_batch_end" if self._optimizer else "on_validation_batch_end",
            )

    def _batch_loop(
        self, dataloader: DataLoader, optimizer: Optional[Optimizer] = None
    ):
        if optimizer:
            self._train_total_batch_loss = 0.0
        else:
            self._validation_total_batch_loss = 0.0

        for batch_index, batch in enumerate(dataloader):
            if optimizer:
                self._train_batch_index = batch_index
            else:
                self._validation_batch_index = batch_index

            inputs: Tensor = batch[0].to(self._device)
            labels: Tensor = batch[1].to(self._device)

            self._process_batch(inputs=inputs, labels=labels, optimizer=optimizer)

            if optimizer:
                self._train_total_batch_loss += self._train_batch_loss  # type: ignore
            else:
                self._validation_total_batch_loss += self._validation_batch_loss  # type: ignore

    def _process_epoch(self):
        if not self._train_dataloader:
            raise  # TODO raise correct error

        self._train_epoch_loss = 0.0

        self._model.train()

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_epoch_start",
            )

        self._batch_loop(dataloader=self._train_dataloader, optimizer=self._optimizer)

        self._train_epoch_loss += self._train_total_batch_loss  # type: ignore

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_epoch_end",
            )

    def _validation_loop(self) -> float:
        if not self._validation_dataloader:
            raise  # TODO raise correct error

        self._validation_loss = 0.0

        self._model.eval()

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_validation_start",
                dataloader=self._validation_dataloader,
                device=self._device,
                model=self._model,
                criterion=self._criterion,
            )

        with torch.no_grad():
            self._validation_loss += self._batch_loop(
                dataloader=self._validation_dataloader
            )  # type: ignore

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_validation_end",
                dataloader=self._validation_dataloader,
                device=self._device,
                model=self._model,
                criterion=self._criterion,
            )

        return self._validation_loss

    def _test_loop(self):
        if not self._test_dataloader:
            raise  # TODO raise correct error

        self._model.eval()

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_test_start",
                dataloader=self._test_dataloader,
                model=self._model,
                device=self._device,
            )

        with torch.no_grad():
            for batch_index, batch in enumerate(self._test_dataloader):
                self._test_batch_index = batch_index

                if self._callbacks:
                    log_callback(
                        self._callbacks,
                        "on_test_batch_start",
                    )

                inputs: Tensor = batch[0].to(self._device)
                labels: Tensor = batch[1].to(self._device)
                predictions: Tensor = self._model(inputs)
                loss = self._criterion(predictions, labels)

                self._test_batch_loss = loss.item()

                if self._callbacks:
                    log_callback(
                        self._callbacks,
                        "on_test_batch_end",
                    )

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_test_end",
            )

    def _train_val_loop(self, num_epochs: int):
        if not self._train_dataloader:
            raise  # TODO raise correct error

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_train_start",
                dataLoader=self._train_dataloader,
            )

        for epoch_num in range(num_epochs):
            self._train_epoch_num = epoch_num

            self._process_epoch()

            if self._validation_dataloader:
                self._validation_loop()

        if self._callbacks:
            log_callback(
                self._callbacks,
                "on_train_end",
                num_epochs=num_epochs,
            )

    def train(self, num_epochs: int):
        self._train_val_loop(num_epochs=num_epochs)

    def test(self):
        self._test_loop()
