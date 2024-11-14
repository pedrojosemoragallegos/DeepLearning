from typing import Optional
import torch
from torch import device as Device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module as Model
from torch.nn.modules.loss import _Loss as Criterion
from torch.cuda.amp import autocast, GradScaler
from .callback.callback_list import CallbackList


def log_callback(callbacks: CallbackList, method: str, **kwargs):
    getattr(callbacks, method)(**kwargs)


def _process_batch(
    inputs: Tensor,
    labels: Tensor,
    optimizer: Optional[Optimizer],
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    scaler: Optional[GradScaler] = None,
) -> float:
    if optimizer:
        optimizer.zero_grad()

    log_callback(
        callbacks,
        "on_train_batch_start" if optimizer else "on_validation_batch_start",
        inputs=inputs,
        labels=labels,
        optimizer=optimizer,
        model=model,
        criterion=criterion,
        scalar=scaler,
    )

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
        "on_train_batch_end" if optimizer else "on_validation_batch_end",
        inputs=inputs,
        labels=labels,
        outputs=outputs,
        loss=loss.item(),
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scalar=scaler,
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
    start_step: int = 0,
) -> float:
    for batch_idx, batch in enumerate(
        dataloader
    ):  # TODO use itertools islice to avoid unnecesary iterations
        if batch_idx < start_step:
            continue

        inputs: Tensor = batch[0].to(device)
        labels: Tensor = batch[1].to(device)

        batch_loss: float = _process_batch(
            inputs, labels, optimizer, model, criterion, callbacks, scaler
        )
        total_loss += batch_loss

    return total_loss


def _process_epoch(
    dataloader: DataLoader,
    device: Device,
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
    optimizer: Optional[Optimizer] = None,
    epoch_num: int = 0,
    scaler: Optional[GradScaler] = None,
    start_step: int = 0,
) -> float:
    model.train() if optimizer else model.eval()

    log_callback(
        callbacks,
        "on_epoch_start",
        dataloader=dataloader,
        epoch_num=epoch_num,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        start_step=start_step,
    )

    total_loss: float = _batch_loop(
        dataloader=dataloader,
        device=device,
        total_loss=0.0,
        model=model,
        criterion=criterion,
        callbacks=callbacks,
        optimizer=optimizer,
        scaler=scaler,
        start_step=start_step,
    )

    log_callback(
        callbacks,
        "on_epoch_end",
        epoch_num=epoch_num,
        device=device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        total_loss=total_loss,
        start_step=start_step,
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

    log_callback(
        callbacks,
        "on_validation_start",
        dataloader=dataloader,
        cumulative_loss=cumulative_loss,
        device=device,
        model=model,
        criterion=criterion,
    )

    with torch.no_grad():
        cumulative_loss = _batch_loop(
            dataloader,
            device,
            cumulative_loss,
            model,
            criterion,
            callbacks,
        )

    log_callback(
        callbacks,
        "on_validation_end",
        dataloader=dataloader,
        cumulative_loss=cumulative_loss,
        device=device,
        model=model,
        criterion=criterion,
    )

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
    start_epoch: int = 0,
    start_step: int = 0,  # type: ignore
    scaler: Optional[GradScaler] = None,
):
    log_callback(
        callbacks,
        "on_train_start",
        dataLoader=train_dataloader,
        num_epochs=num_epochs,
        epoch_num=start_epoch,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        validation_dataloader=validation_dataloader,
    )

    for epoch in range(start_epoch, num_epochs):
        _process_epoch(
            train_dataloader,
            device,
            model,
            criterion,
            callbacks,
            optimizer=optimizer,
            epoch_num=epoch,
            scaler=scaler,
            start_step=start_step if epoch == start_epoch else 0,
        )

        start_step: int = 0

        if validation_dataloader:
            validation_loop(
                dataloader=validation_dataloader,
                device=device,
                model=model,
                criterion=criterion,
                callbacks=callbacks,
            )

    log_callback(
        callbacks,
        "on_train_end",
        num_epochs=num_epochs,
        epoch_num=start_epoch,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=train_dataloader,
    )


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
                callbacks,
                "on_test_batch_start",
                batch_index=batch_idx,
                batch=batch,
                dataloader=testloader,
                model=model,
                device=device,
            )

            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            predictions: Tensor = model(inputs)

            log_callback(
                callbacks,
                "on_test_batch_end",
                batch_index=batch_idx,
                batch=batch,
                dataloader=testloader,
                model=model,
                device=device,
                labels=labels,
                predictions=predictions,
            )

    log_callback(
        callbacks,
        "on_test_end",
        model=model,
        dataloader=testloader,
        device=device,
    )
