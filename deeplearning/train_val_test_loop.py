from typing import Optional
import torch
from tqdm.auto import tqdm
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
    PHASE: str = "train" if optimizer else "validation"

    # callback: before forward pass
    log_callback(callbacks, "on_batch_start", phase=PHASE, inputs=inputs, labels=labels)

    if optimizer:
        optimizer.zero_grad()  # training

    outputs: Tensor = model(inputs)  # forward pass

    # callback: after forward pass
    log_callback(callbacks, "on_batch_end", phase=PHASE, outputs=outputs)

    loss: Tensor = criterion(outputs, labels)  # compute loss

    # callback: after loss computation
    log_callback(
        callbacks,
        "on_batch_end",
        phase=PHASE,
        loss=loss.item(),
        outputs=outputs,
        labels=labels,
    )

    if optimizer:
        # callback: before backward pass
        log_callback(callbacks, "on_batch_start", phase="backward", loss=loss.item())

        loss.backward()  # backward pass

        # callback: after backward pass
        log_callback(callbacks, "on_batch_end", phase="backward_done", model=model)

        optimizer.step()  # optimizer step

        # callback: after optimizer step
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
    show_progress: bool = False,
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
        show_progress (bool): Whether to display a progress bar.

    Returns:
        float: Total loss for the DataLoader.
    """

    # initialize the tqdm progress bar if requested
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

    # close tqdm progress bar if used
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
) -> float:
    """
    Process an entire epoch of data.

    Args:
        dataloader (DataLoader): DataLoader for the current phase.
        device (Device): Device to process the data on.
        model (Model): Neural network model.
        criterion (Criterion): Loss function.
        callbacks (CallbackList): List of callbacks.
        optimizer (Optional[Optimizer]): Optimizer for training phase.
        show_progress (bool): Whether to display a progress bar.

    Returns:
        float: Total loss for the epoch.
    """
    PHASE: str = "train" if optimizer else "validation"

    # callback: start of epoch
    log_callback(callbacks, "on_epoch_start", phase=PHASE)

    model.train() if optimizer else model.eval()  # Set model mode

    total_loss = 0.0

    with torch.set_grad_enabled(PHASE == "train"):
        total_loss = _batch_loop(
            dataloader,
            device,
            total_loss,
            model,
            criterion,
            callbacks,
            optimizer,
            show_progress,
        )

    # callback: end of epoch
    log_callback(callbacks, "on_epoch_end", phase=PHASE, total_loss=total_loss)

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
    """
    Perform a validation loop over the given DataLoader.

    Args:
        dataloader (DataLoader): DataLoader for validation data.
        device (Device): Device to process the data on.
        model (Model): Neural network model.
        criterion (Criterion): Loss function.
        callbacks (CallbackList): List of callbacks.
        cumulative_loss (float): Initial cumulative loss.
        show_progress (bool): Whether to display a progress bar.

    Returns:
        float: Cumulative loss over the validation data.
    """
    model.eval()  # Set model to evaluation mode

    # callback: start of validation
    log_callback(callbacks, "on_validation_start", cumulative_loss=cumulative_loss)

    with torch.no_grad():  # disable gradient calculation for validation
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

    # callback: end of validation
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
) -> None:
    """
    Train the model for a specified number of epochs, with optional validation.

    Args:
        train_dataloader (DataLoader): DataLoader for training data.
        device (Device): Device to process the data on.
        model (Model): Neural network model.
        criterion (Criterion): Loss function.
        optimizer (Optimizer): Optimizer for training.
        callbacks (CallbackList): List of callbacks.
        num_epochs (int): Number of epochs to train for.
        validation_dataloader (Optional[DataLoader]): DataLoader for validation data.
        show_progress (bool): Whether to display a progress bar.
    """
    cumulative_loss: float = 0.0

    # callback: start of training
    log_callback(
        callbacks,
        "on_train_start",
        num_epochs=num_epochs,
        cumulative_loss=cumulative_loss,
    )

    for epoch in range(num_epochs):
        # training phase
        train_loss = _process_epoch(
            dataloader=train_dataloader,
            device=device,
            model=model,
            criterion=criterion,
            callbacks=callbacks,
            optimizer=optimizer,
            show_progress=show_progress,
        )

        cumulative_loss += train_loss

        # validation phase (if applicable)
        if validation_dataloader:
            validation_loop(
                dataloader=validation_dataloader,
                device=device,
                model=model,
                criterion=criterion,
                callbacks=callbacks,
                cumulative_loss=0.0,
                show_progress=show_progress,
            )

    # callback: end of training
    log_callback(callbacks, "on_train_end", cumulative_loss=cumulative_loss)


def test_loop(
    testloader: DataLoader,
    model: Model,
    device: Device,
    callbacks: CallbackList = CallbackList(),
    show_progress: bool = False,
):
    """
    Loop for testing.

    Args:
        testloader (DataLoader): DataLoader for testing.
        model (Model): Neural network model.
        device (Device): Device to process data on.
        callbacks (CallbackList): List of callbacks.
        show_progress (bool): Whether to display a progress bar.
    """
    model.eval()

    # callback: start of testing
    log_callback(
        callbacks, "on_test_start", dataloader=testloader, model=model, device=device
    )

    # Optional progress bar for testing
    testloader_iter = (
        tqdm(testloader, desc="Testing", leave=False) if show_progress else testloader
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader_iter):
            # callback: start of test batch
            log_callback(callbacks, "on_batch_start", batch_idx=batch_idx, batch=batch)

            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            predictions: Tensor = model(inputs)

            # callback: end of test batch
            log_callback(
                callbacks,
                "on_batch_end",
                batch_idx=batch_idx,
                inputs=inputs,
                labels=labels,
                predictions=predictions,
            )

    # close tqdm progress bar if used
    if show_progress and hasattr(testloader_iter, "close"):
        testloader_iter.close()

    # callback: end of testing
    log_callback(callbacks, "on_test_end", model=model, num_batches=len(testloader))
