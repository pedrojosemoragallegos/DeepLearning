from typing import Optional, Tuple
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
    if optimizer:
        optimizer.zero_grad()  # train

    outputs: Tensor = model(inputs)  # train | validation
    loss: Tensor = criterion(outputs, labels)  # train | validation

    if optimizer:
        loss.backward()  # train
        optimizer.step()  # train

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
    for batch in dataloader:
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
    return total_loss


def train_val_loop(
    is_train: bool,
    num_epochs: Optional[int],  # train
    dataloader: DataLoader,  # train | validation
    device: Device,  # train | validation
    optimizer: Optional[Optimizer],  # train
    model: Model,  # train | validation
    criterion: Criterion,  # train | validation
    start_epoch: int = 0,  # Resuming from a specific epoch
    start_epoch_loss: float = 0.0,  # Resuming cumulative loss
    cumulative_loss: float = 0.0,  # Resuming cumulative loss across epochs
):
    if is_train:
        for epoch in range(num_epochs):
            epoch_loss: float = start_epoch_loss
            epoch_loss: float = _batch_loop(
                dataloader=dataloader,
                device=device,
                loss=epoch_loss,
                optimizer=optimizer,
                model=model,
                criterion=criterion,
            )
            cumulative_loss += epoch_loss
    else:
        model.eval()

        with torch.no_grad():
            loss: float = start_epoch_loss
            loss = _batch_loop(
                dataloader=dataloader,
                device=device,
                loss=loss,
                model=model,
                criterion=criterion,
            )


def test_loop(
    testloader: DataLoader[Tuple[Tensor, Tensor]],
    model: Model,
    device: Device,
):
    model.eval()

    with torch.no_grad():
        for batch in testloader:
            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            prediction: Tensor = model(inputs)
