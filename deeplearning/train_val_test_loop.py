from typing import Optional
import torch
from torch import device as Device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module as Model
from torch.nn.modules.loss import _Loss as Criterion


def _process_batch(
    inputs: Tensor,  # train | validation
    labels: Tensor,  # train | validation
    optimizer: Optional[Optimizer],  # train
    model: Model,  # train | validation
    criterion: Criterion,  # train | validation
) -> float:
    # callback: before forward pass
    if optimizer:
        optimizer.zero_grad()  # train

    outputs: Tensor = model(inputs)  # train | validation
    # callback: after forward pass, before loss computation

    loss: Tensor = criterion(outputs, labels)  # train | validation
    # callback: after loss computation

    if optimizer:
        # callback: before backward pass
        loss.backward()  # train
        # callback: after backward pass, before optimizer step
        optimizer.step()  # train
        # callback: after optimizer step

    batch_loss: float = loss.item()  # train | validation
    return batch_loss


def _batch_loop(
    dataloader: DataLoader,
    device: Device,
    total_loss: float,
    model: Model,  # train | validation
    criterion: Criterion,  # train | validation
    optimizer: Optional[Optimizer] = None,  # train
) -> float:
    # callback: start batch_loop
    for batch in dataloader:
        # callback: start batch
        inputs: Tensor = batch[0].to(device)
        labels: Tensor = batch[1].to(device)

        batch_loss: float = _process_batch(
            inputs=inputs,
            labels=labels,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
        )

        total_loss += batch_loss
        # callback: end batch
    # callback: end batch_loop
    return total_loss


def train_val_loop(
    is_train: bool,
    num_epochs: Optional[int],  # train
    dataloader: DataLoader,  # train | validation
    device: Device,  # train | validation
    optimizer: Optional[Optimizer],  # train
    model: Model,  # train | validation
    criterion: Criterion,  # train | validation
    cumulative_loss: float = 0.0,  # resuming cumulative loss
):
    if is_train:
        # callback: start train_loop
        for epoch in range(num_epochs):
            # callback: start epoch
            epoch_loss: float = 0.0  # reseting loss for each epoch
            epoch_loss: float = _batch_loop(
                dataloader=dataloader,
                device=device,
                loss=epoch_loss,
                optimizer=optimizer,
                model=model,
                criterion=criterion,
            )
            cumulative_loss += epoch_loss
            # callback: end epoch
        # callback: end train_loop
    else:
        model.eval()

        with torch.no_grad():
            # callback: start validation
            loss: float = cumulative_loss
            loss = _batch_loop(
                dataloader=dataloader,
                device=device,
                loss=loss,
                model=model,
                criterion=criterion,
            )
            # callback: end validation


def test_loop(
    testloader: DataLoader,
    model: Model,
    device: Device,
):
    model.eval()

    with torch.no_grad():
        # callback: start testing
        for batch in testloader:
            # callback: start test_batch
            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            prediction: Tensor = model(inputs)
            # callback: end test_batch
        # callback: end testing
