from typing import Optional
import torch
from tqdm.auto import tqdm
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
    show_progress: bool = False,
) -> float:
    dataloader_iter = (
        tqdm(dataloader, desc="Batch Progress", leave=False)
        if show_progress
        else dataloader
    )

    for batch_idx, batch in enumerate(dataloader_iter):
        inputs: Tensor = batch[0].to(device)
        labels: Tensor = batch[1].to(device)

        batch_loss: float = _process_batch(
            inputs, labels, optimizer, model, criterion, callbacks
        )
        total_loss += batch_loss

    if show_progress and hasattr(dataloader_iter, "close"):
        dataloader_iter.close()

    return total_loss


def _process_epoch(
    dataloader: DataLoader,
    device: Device,
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    optimizer: Optional[Optimizer] = None,
    show_progress: bool = False,
    epoch: int = 0,
    scaler: Optional[GradScaler] = None,
) -> float:
    PHASE: str = "train" if optimizer else "validation"

    log_callback(callbacks, "on_epoch_start", phase=PHASE, epoch=epoch)
    model.train() if optimizer else model.eval()

    total_loss: float = 0.0
    dataloader_iter = (
        tqdm(dataloader, desc=f"{PHASE.capitalize()} Epoch {epoch}", leave=False)
        if show_progress
        else dataloader
    )

    try:
        for batch_idx, batch in enumerate(dataloader_iter):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            batch_loss = _process_batch(
                inputs, labels, optimizer, model, criterion, callbacks, scaler=scaler
            )
            total_loss += batch_loss

    except KeyboardInterrupt:
        log_callback(callbacks, "on_training_stopped", epoch=epoch, batch_idx=batch_idx)
        raise StopTrainingException("Training interrupted manually.")

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
    show_progress: bool = False,
) -> float:
    model.eval()

    log_callback(callbacks, "on_validation_start", cumulative_loss=cumulative_loss)

    with torch.no_grad():
        dataloader_iter = (
            tqdm(dataloader, desc="Validation", leave=False)
            if show_progress
            else dataloader
        )

        cumulative_loss = _batch_loop(
            dataloader_iter, device, cumulative_loss, model, criterion, callbacks
        )

        if show_progress and hasattr(dataloader_iter, "close"):
            dataloader_iter.close()

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
    show_progress: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    use_mixed_precision: bool = False,
) -> None:
    start_epoch: int = 0
    scaler = GradScaler() if use_mixed_precision else None

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
            show_progress=show_progress,
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
                show_progress=show_progress,
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
    show_progress: bool = False,
):
    model.eval()

    log_callback(
        callbacks, "on_test_start", dataloader=testloader, model=model, device=device
    )

    testloader_iter = (
        tqdm(testloader, desc="Testing", leave=False) if show_progress else testloader
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader_iter):
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

    if show_progress and hasattr(testloader_iter, "close"):
        testloader_iter.close()

    log_callback(callbacks, "on_test_end", model=model, num_batches=len(testloader))
