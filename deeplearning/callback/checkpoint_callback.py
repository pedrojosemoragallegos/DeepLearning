from pathlib import Path
from typing import Union, Optional, Any
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from .callback import Callback
from deeplearning.utils.checkpoint_handler import save_checkpoint


class CheckpointCallback(Callback):
    def __init__(self, save_path: Union[Path, str]) -> None:
        self._save_path: Path = Path(save_path)
        self._save_path.mkdir(parents=True, exist_ok=True)

    def on_train_interrupt(self, **kwargs: Any) -> None:
        epoch_num: int = kwargs.get("epoch_num")  # type: ignore
        model: Module = kwargs.get("model")  # type: ignore
        optimizer: Optimizer = kwargs.get("optimizer")  # type: ignore
        train_epoch_loss: float = kwargs.get("train_epoch_loss")  # type: ignore
        scheduler: Optional[_LRScheduler] = kwargs.get("scheduler")  # type: ignore

        checkpoint_path: Path = (
            self._save_path / f"checkpoint_interrupt_epoch_{epoch_num}.pth"
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch_num,
            loss=train_epoch_loss,
            filepath=checkpoint_path,
        )
        print(f"Training interrupted. Checkpoint saved at {checkpoint_path}")
