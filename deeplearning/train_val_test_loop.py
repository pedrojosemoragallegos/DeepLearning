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


class StopTrainingException(Exception):
    pass


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
        model=model,
        optimizer=optimizer,
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
) -> float:
    for batch_idx, batch in enumerate(dataloader):
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
                callbacks,
                "on_checkpoint_load",
                success=True,
                scaler=scaler,
                epoch_num=start_epoch,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                dataloader=validation_dataloader,
                resume_from_checkpoint=resume_from_checkpoint,
                use_mixed_precision=use_mixed_precision,
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
        resume_from_checkpoint=resume_from_checkpoint,
        use_mixed_precision=use_mixed_precision,
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
        )

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
            "on_checkpoint_save",
            num_epochs=num_epochs,
            epoch_num=start_epoch,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=validation_dataloader,
            resume_from_checkpoint=resume_from_checkpoint,
            use_mixed_precision=use_mixed_precision,
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
        resume_from_checkpoint=resume_from_checkpoint,
        use_mixed_precision=use_mixed_precision,
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
        batch_index=batch_idx,
        batch=batch,
        dataloader=testloader,
        device=device,
        labels=labels,
        predictions=predictions,
    )
