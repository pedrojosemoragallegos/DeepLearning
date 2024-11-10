from typing import Optional
import torch
from torch import device as Device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module as Model
from torch.nn.modules.loss import _Loss as Criterion

from deeplearning.callback import CallbackList


def log_callback(callbacks: CallbackList, method: str, **kwargs):
    """
    Dynamically call a callback method with provided logs.

    Args:
        callbacks (CallbackList): List of callbacks.
        method (str): Callback method to call (e.g., 'on_batch_start').
        **kwargs: Additional logs to pass to the callback.
    """
    getattr(callbacks, method)(logs=kwargs)


def _process_batch(
    inputs: Tensor,
    labels: Tensor,
    optimizer: Optional[Optimizer],
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
) -> float:
    """
    Process a single batch of data.

    Args:
        inputs (Tensor): Input data.
        labels (Tensor): Ground truth labels.
        optimizer (Optional[Optimizer]): Optimizer, if in training mode.
        model (Model): Neural network model.
        criterion (Criterion): Loss function.
        callbacks (CallbackList): List of callbacks.

    Returns:
        float: Loss value for the batch.
    """
    phase: str = "train" if optimizer else "validation"

    # Callback: before forward pass
    log_callback(callbacks, "on_batch_start", phase=phase, inputs=inputs, labels=labels)

    if optimizer:
        optimizer.zero_grad()  # Training

    outputs: Tensor = model(inputs)  # Forward pass

    # Callback: after forward pass
    log_callback(callbacks, "on_batch_end", phase=phase, outputs=outputs)

    loss: Tensor = criterion(outputs, labels)  # Compute loss

    # Callback: after loss computation
    log_callback(
        callbacks,
        "on_batch_end",
        phase=phase,
        loss=loss.item(),
        outputs=outputs,
        labels=labels,
    )

    if optimizer:
        # Callback: before backward pass
        log_callback(callbacks, "on_batch_start", phase="backward", loss=loss.item())

        loss.backward()  # Backward pass

        # Callback: after backward pass
        log_callback(callbacks, "on_batch_end", phase="backward_done", model=model)

        optimizer.step()  # Optimizer step

        # Callback: after optimizer step
        log_callback(
            callbacks,
            "on_batch_end",
            phase="step_done",
            updated_parameters=[p for p in model.parameters() if p.grad is not None],
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
) -> float:
    """
    Process an entire DataLoader in a single loop.

    Args:
        dataloader (DataLoader): DataLoader for the current phase.
        device (Device): Device to process the data on.
        total_loss (float): Initial loss (cumulative across epochs).
        model (Model): Neural network model.
        criterion (Criterion): Loss function.
        callbacks (CallbackList): List of callbacks.
        optimizer (Optional[Optimizer]): Optimizer for training.

    Returns:
        float: Total loss for the DataLoader.
    """

    # Callback: start of batch loop
    log_callback(
        callbacks, "on_batch_start", phase="batch_loop_start", total_loss=total_loss
    )

    for batch_idx, batch in enumerate(dataloader):
        # Callback: start of individual batch
        log_callback(callbacks, "on_batch_start", batch_idx=batch_idx, batch=batch)

        inputs: Tensor = batch[0].to(device)
        labels: Tensor = batch[1].to(device)

        batch_loss: float = _process_batch(
            inputs, labels, optimizer, model, criterion, callbacks
        )
        total_loss += batch_loss

        # Callback: end of individual batch
        log_callback(
            callbacks,
            "on_batch_end",
            batch_idx=batch_idx,
            batch_loss=batch_loss,
            total_loss=total_loss,
        )

    # Callback: end of batch loop
    log_callback(
        callbacks, "on_batch_end", phase="batch_loop_end", total_loss=total_loss
    )

    return total_loss


def train_val_loop(
    is_train: bool,
    num_epochs: Optional[int],
    dataloader: DataLoader,
    device: Device,
    optimizer: Optional[Optimizer],
    model: Model,
    criterion: Criterion,
    cumulative_loss: float = 0.0,
    callbacks: CallbackList = CallbackList(),
):
    """
    Loop for training or validation.

    Args:
        is_train (bool): Whether to perform training or validation.
        num_epochs (Optional[int]): Number of epochs (training only).
        dataloader (DataLoader): DataLoader for the phase.
        device (Device): Device to process data on.
        optimizer (Optional[Optimizer]): Optimizer for training.
        model (Model): Neural network model.
        criterion (Criterion): Loss function.
        cumulative_loss (float): Initial cumulative loss.
        callbacks (CallbackList): List of callbacks.
    """
    if is_train:
        # Callback: start of training
        log_callback(
            callbacks,
            "on_train_start",
            num_epochs=num_epochs,
            cumulative_loss=cumulative_loss,
        )

        for epoch in range(num_epochs):
            # Callback: start of epoch
            log_callback(
                callbacks,
                "on_epoch_start",
                epoch=epoch,
                cumulative_loss=cumulative_loss,
            )

            epoch_loss = _batch_loop(
                dataloader, device, 0.0, model, criterion, callbacks, optimizer
            )
            cumulative_loss += epoch_loss

            # Callback: end of epoch
            log_callback(
                callbacks,
                "on_epoch_end",
                epoch=epoch,
                epoch_loss=epoch_loss,
                cumulative_loss=cumulative_loss,
            )

        # Callback: end of training
        log_callback(callbacks, "on_train_end", cumulative_loss=cumulative_loss)
    else:
        model.eval()

        # Callback: start of validation
        log_callback(callbacks, "on_validation_start", cumulative_loss=cumulative_loss)

        with torch.no_grad():
            cumulative_loss = _batch_loop(
                dataloader, device, cumulative_loss, model, criterion, callbacks
            )

        # Callback: end of validation
        log_callback(callbacks, "on_validation_end", cumulative_loss=cumulative_loss)


def test_loop(
    testloader: DataLoader,
    model: Model,
    device: Device,
    callbacks: CallbackList = CallbackList(),
):
    """
    Loop for testing.

    Args:
        testloader (DataLoader): DataLoader for testing.
        model (Model): Neural network model.
        device (Device): Device to process data on.
        callbacks (CallbackList): List of callbacks.
    """
    model.eval()

    # Callback: start of testing
    log_callback(
        callbacks, "on_test_start", dataloader=testloader, model=model, device=device
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            # Callback: start of test batch
            log_callback(callbacks, "on_batch_start", batch_idx=batch_idx, batch=batch)

            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            predictions: Tensor = model(inputs)

            # Callback: end of test batch
            log_callback(
                callbacks,
                "on_batch_end",
                batch_idx=batch_idx,
                inputs=inputs,
                labels=labels,
                predictions=predictions,
            )

    # Callback: end of testing
    log_callback(callbacks, "on_test_end", model=model, num_batches=len(testloader))
