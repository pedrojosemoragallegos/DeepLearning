from tqdm.autonotebook import tqdm
from typing import Optional, Dict, Any, cast
from .callback import Callback
from torch.utils.data import DataLoader


class ProgressBarCallback(Callback):
    def __init__(self):
        self._train_progress_bar: Optional[tqdm] = None
        self._batch_progress_bar: Optional[tqdm] = None
        self._validation_progress_bar: Optional[tqdm] = None

    def on_train_start(self, **kwargs: Dict[str, Any]):
        num_epochs: int = kwargs.get("num_epochs")  # type: ignore

        self._train_progress_bar = tqdm(
            total=num_epochs,
            desc="Training Progress",
            position=0,
            leave=True,
            unit="epoch",
        )

    def on_train_epoch_start(self, **kwargs: Dict[str, Any]):
        dataloader: DataLoader = cast(DataLoader, kwargs.get("dataloader"))
        epoch_num: int = cast(int, kwargs.get("epoch_num"))
        total_batches: int = len(dataloader)

        if self._batch_progress_bar:
            self._batch_progress_bar.close()  # type: ignore
            self._batch_progress_bar = None

        self._batch_progress_bar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch_num + 1}",
            position=1,
            leave=False,
            unit="batch",
        )

    def on_train_batch_end(self, **kwargs: Dict[str, Any]):
        self._batch_progress_bar.update(1)  # type: ignore

    def on_train_epoch_end(self, **kwargs: Dict[str, Any]):
        self._batch_progress_bar.close()  # type: ignore
        self._batch_progress_bar = None
        self._train_progress_bar.update(1)  # type: ignore

    def on_validation_start(self, **kwargs: Dict[str, Any]):
        dataloader: DataLoader = kwargs.get("dataloader")  # type: ignore
        total_batches: int = len(dataloader)

        self._validation_progress_bar = tqdm(
            total=total_batches,
            desc="Validation Batches",
            position=2,
            leave=False,
            unit="batch",
        )

    def on_validation_batch_end(self, **kwargs: Dict[str, Any]):
        self._validation_progress_bar.update(1)  # type: ignore

    def on_validation_end(self, **kwargs: Dict[str, Any]):
        self._validation_progress_bar.close()  # type: ignore
        self._validation_progress_bar = None

    def on_train_end(self, **kwargs: Dict[str, Any]):
        self._train_progress_bar.close()  # type: ignore
        self._train_progress_bar = None
