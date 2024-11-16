from typing import Dict, Any
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
    filepath: Union[Path, str],
):
    filepath = Path(filepath)

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


def load_checkpoint(
    model: Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    filepath: Union[Path, str],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Union[int, float]]:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file {filepath} not found.")

    checkpoint: Dict[str, Any] = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"])

    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
    }
