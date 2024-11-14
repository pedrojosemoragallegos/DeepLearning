from pathlib import Path
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Union


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    loss: float,
    filepath: Union[Path, str],  # type: ignore
):
    filepath: Path = Path(filepath)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint["rng_state"] = torch.get_rng_state()

    torch.save(checkpoint, filepath)
