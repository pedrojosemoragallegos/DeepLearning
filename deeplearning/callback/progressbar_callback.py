from tqdm.auto import tqdm
from typing import Optional, Dict, Any
from .callback import Callback


class ProgressBarCallback(Callback):
    def __init__(self) -> None:
        self._train_progress_bar: Optional[tqdm] = None
        self._batch_progress_bar: Optional[tqdm] = None
        self._validation_progress_bar: Optional[tqdm] = None

    def on_train_start(self, **kwargs: Dict[str, Any]) -> None:
        num_epochs = kwargs.get("num_epochs")

        assert (
            num_epochs is not None
        ), "Expected 'num_epochs' in kwargs for on_train_start"

        self._train_progress_bar = tqdm(
            total=num_epochs,
            desc="Training Progress",
            position=0,
            leave=True,
            unit="epoch",
        )

    def on_epoch_start(self, **kwargs: Dict[str, Any]) -> None:
        dataloader = kwargs.get("dataloader")
        epoch_num = kwargs.get("epoch_num")

        total_batches = len(dataloader)

        assert total_batches > 0, "Dataloader has no batches!"

        if self._batch_progress_bar:
            self._batch_progress_bar.close()

        self._batch_progress_bar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch_num + 1}",
            position=1,
            leave=False,
            unit="batch",
        )

    def on_epoch_end(self, **kwargs: Dict[str, Any]) -> None:
        if self._batch_progress_bar:
            self._batch_progress_bar.close()

        if self._train_progress_bar:
            self._train_progress_bar.update(1)

    def on_train_batch_end(self, **kwargs: Dict[str, Any]) -> None:
        if self._batch_progress_bar:
            self._batch_progress_bar.update(1)

    def on_validation_start(self, **kwargs: Dict[str, Any]) -> None:
        dataloader = kwargs.get("dataloader")

        assert (
            dataloader is not None
        ), "Expected 'dataloader' in kwargs for on_validation_start"

        total_batches = len(dataloader)
        if self._validation_progress_bar:
            self._validation_progress_bar.close()

        self._validation_progress_bar = tqdm(
            total=total_batches,
            desc="Validation Batches",
            position=2,
            leave=False,
            unit="batch",
        )

    def on_validation_batch_end(self, **kwargs: Dict[str, Any]) -> None:
        if self._validation_progress_bar:
            self._validation_progress_bar.update(1)

    def on_validation_end(self, **kwargs: Dict[str, Any]) -> None:
        if self._validation_progress_bar:
            self._validation_progress_bar.close()

    def on_train_end(self, **kwargs: Dict[str, Any]) -> None:
        if self._train_progress_bar:
            self._train_progress_bar.close()
