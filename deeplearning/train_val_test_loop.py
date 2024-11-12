from typing import Optional
import torch
from torch import device as Device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module as Model
from torch.nn.modules.loss import _Loss as Criterion
from torch.cuda.amp import autocast, GradScaler
from .callback_list import CallbackList


class StopTrainingException(Exception):
    pass


def log_callback(callbacks: CallbackList, method: str, **kwargs):
    getattr(callbacks, method)(logs=kwargs)


def _process_batch(
    inputs: Tensor,
    labels: Tensor,
    optimizer: Optional[Optimizer],
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    scaler: Optional[GradScaler] = None,
) -> float:
    PHASE: str = "train" if optimizer else "validation"

    log_callback(callbacks, "on_batch_start", phase=PHASE, inputs=inputs, labels=labels)

    if optimizer:
        optimizer.zero_grad()

    with autocast(enabled=scaler is not None):
        outputs: Tensor = model(inputs)
        loss: Tensor = criterion(outputs, labels)

    if optimizer:
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        log_callback(
            callbacks,
            "on_optimizer_step",
            phase=PHASE,
            updated_parameters=[p for p in model.parameters() if p.grad is not None],
        )

    log_callback(
        callbacks,
        "on_batch_end",
        phase=PHASE,
        loss=loss.item(),
        outputs=outputs,
        labels=labels,
    )

    return loss.item()


def _batch_loop(
    dataloader: DataLoader,
    device: Device,
    total_loss: float,
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
) -> float:
    PHASE: str = "train" if optimizer else "validation"

    for batch_idx, batch in enumerate(dataloader):
        inputs: Tensor = batch[0].to(device)
        labels: Tensor = batch[1].to(device)

        batch_loss: float = _process_batch(
            inputs, labels, optimizer, model, criterion, callbacks, scaler
        )
        total_loss += batch_loss

        log_callback(
            callbacks,
            "on_batch_start",
            phase=PHASE,
            batch_idx=batch_idx,
            loss=batch_loss,
            total_batches=len(dataloader),
        )

    return total_loss


def _process_epoch(
    dataloader: DataLoader,
    device: Device,
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    optimizer: Optional[Optimizer] = None,
    epoch: int = 0,
    scaler: Optional[GradScaler] = None,
) -> float:
    PHASE: str = "train" if optimizer else "validation"

    log_callback(
        callbacks,
        "on_epoch_start",
        phase=PHASE,
        epoch=epoch,
        total_batches=len(dataloader),
    )

    model.train() if optimizer else model.eval()

    total_loss: float = _batch_loop(
        dataloader,
        device,
        0.0,
        model,
        criterion,
        callbacks,
        optimizer=optimizer,
        scaler=scaler,
    )

    log_callback(
        callbacks, "on_epoch_end", phase=PHASE, total_loss=total_loss, epoch=epoch
    )

    return total_loss


def validation_loop(
    dataloader: DataLoader,
    device: Device,
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    cumulative_loss: float = 0.0,
) -> float:
    model.eval()

    log_callback(callbacks, "on_validation_start", cumulative_loss=cumulative_loss)

    with torch.no_grad():
        cumulative_loss = _batch_loop(
            dataloader,
            device,
            cumulative_loss,
            model,
            criterion,
            callbacks,
            optimizer=None,
            scaler=None,
        )

    log_callback(callbacks, "on_validation_end", cumulative_loss=cumulative_loss)

    return cumulative_loss


def train_val_loop(
    train_dataloader: DataLoader,
    device: Device,
    model: Model,
    criterion: Criterion,
    optimizer: Optimizer,
    callbacks: CallbackList,
    num_epochs: int,
    validation_dataloader: Optional[DataLoader] = None,
    resume_from_checkpoint: Optional[str] = None,
    use_mixed_precision: bool = False,
) -> None:
    start_epoch: int = 0
    scaler: GradScaler = GradScaler() if use_mixed_precision else None

    if resume_from_checkpoint:
        try:
            checkpoint = torch.load(resume_from_checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

            if scaler and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            log_callback(
                callbacks, "on_checkpoint_load", success=True, epoch=start_epoch
            )
        except FileNotFoundError:
            log_callback(
                callbacks,
                "on_checkpoint_load",
                success=False,
                message="No checkpoint found.",
            )
        except KeyError as e:
            log_callback(
                callbacks,
                "on_checkpoint_load",
                success=False,
                message=f"Checkpoint missing key: {e}",
            )
            raise KeyError(f"Checkpoint missing key: {e}")
        except Exception as e:
            log_callback(
                callbacks,
                "on_checkpoint_load",
                success=False,
                message=f"Failed to load checkpoint: {e}",
            )
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    log_callback(callbacks, "on_train_start", num_epochs=num_epochs)

    for epoch in range(start_epoch, num_epochs):
        _process_epoch(
            train_dataloader,
            device,
            model,
            criterion,
            callbacks,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
        )

        if validation_dataloader:
            _process_epoch(
                validation_dataloader,
                device,
                model,
                criterion,
                callbacks,
                optimizer=None,
                epoch=epoch,
                scaler=None,
            )

        log_callback(
            callbacks,
            "on_checkpoint_save",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scaler_state_dict=(scaler.state_dict() if scaler else None),
        )

    log_callback(callbacks, "on_train_end")


def test_loop(
    testloader: DataLoader,
    model: Model,
    device: Device,
    callbacks: CallbackList = CallbackList(),
):
    model.eval()

    log_callback(
        callbacks, "on_test_start", dataloader=testloader, model=model, device=device
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            log_callback(
                callbacks, "on_test_batch_start", batch_idx=batch_idx, batch=batch
            )

            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            predictions: Tensor = model(inputs)

            log_callback(
                callbacks,
                "on_test_batch_end",
                batch_idx=batch_idx,
                inputs=inputs,
                labels=labels,
                predictions=predictions,
            )

    log_callback(callbacks, "on_test_end", model=model, num_batches=len(testloader))
