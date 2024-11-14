from pathlib import Path
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Union, cast
from .callback import Callback
from ..checkpoint_handler import save_checkpoint


class CheckpointCallback(Callback):
    def __init__(self, filepath: Union[Path, str]) -> None:
        self._filepath = Path(filepath)

    def on_train_end(
        self,
        **kwargs,
    ) -> None:
        save_checkpoint(
            filepath=self._filepath,
            model=cast(Module, kwargs.get("model")),
            optimizer=cast(Optimizer, kwargs.get("optimizer")),
            scheduler=cast(Optional[_LRScheduler], kwargs.get("scheduler")),
            epoch=cast(int, kwargs.get("epoch")),
            loss=cast(float, kwargs.get("loss")),
        )
