from typing import Optional
import torch
from torch import device as Device
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module as Model
from torch.nn.modules.loss import _Loss as Criterion

from deeplearning.callback import CallbackList


def _process_batch(
    inputs: Tensor,
    labels: Tensor,
    optimizer: Optional[Optimizer],
    model: Model,
    criterion: Criterion,
    callbacks: CallbackList,
) -> float:
    # callback: before forward pass
    callbacks.on_batch_start(
        batch=None,
        logs={
            "phase": "train" if optimizer else "validation",
            "inputs": inputs,
            "labels": labels,
        },
    )

    if optimizer:
        optimizer.zero_grad()  # train

    outputs: Tensor = model(inputs)  # train | validation

    # callback: after forward pass, before loss computation
    callbacks.on_batch_end(batch=None, logs={"outputs": outputs})

    loss: Tensor = criterion(outputs, labels)  # train | validation

    # callback: after loss computation
    callbacks.on_batch_end(
        batch=None, logs={"loss": loss.item(), "outputs": outputs, "labels": labels}
    )

    if optimizer:
        # callback: before backward pass
        callbacks.on_batch_start(
            batch=None, logs={"phase": "backward", "loss": loss.item()}
        )
        loss.backward()  # train

        # callback: after backward pass, before optimizer step
        callbacks.on_batch_end(
            batch=None,
            logs={
                "phase": "backward_done",
                "gradients": [p.grad for p in model.parameters() if p.grad is not None],
            },
        )

        optimizer.step()  # train

        # callback: after optimizer step
        callbacks.on_batch_end(
            batch=None,
            logs={
                "phase": "step_done",
                "updated_parameters": [p for p in model.parameters()],
            },
        )

    batch_loss: float = loss.item()  # train | validation
    return batch_loss


def _batch_loop(
    dataloader: DataLoader,
    device: Device,
    total_loss: float,
    model: Model,  # train | validation
    criterion: Criterion,  # train | validation
    callbacks: CallbackList,
    optimizer: Optional[Optimizer] = None,  # train
) -> float:
    # callback: start batch_loop
    callbacks.on_batch_start(
        batch=None, logs={"phase": "batch_loop_start", "total_loss_start": total_loss}
    )

    for batch_idx, batch in enumerate(dataloader):
        # callback: start batch
        callbacks.on_batch_start(
            batch=batch_idx, logs={"batch_idx": batch_idx, "batch_data": batch}
        )

        inputs: Tensor = batch[0].to(device)
        labels: Tensor = batch[1].to(device)

        batch_loss: float = _process_batch(
            inputs=inputs,
            labels=labels,
            optimizer=optimizer,
            model=model,
            criterion=criterion,
            callbacks=callbacks,
        )

        total_loss += batch_loss

        # callback: end batch
        callbacks.on_batch_end(
            batch=batch_idx,
            logs={
                "batch_idx": batch_idx,
                "batch_loss": batch_loss,
                "total_loss_so_far": total_loss,
                "inputs": inputs,
                "labels": labels,
            },
        )

    # callback: end batch_loop
    callbacks.on_batch_end(
        batch=None, logs={"phase": "batch_loop_end", "final_total_loss": total_loss}
    )
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
    callbacks: CallbackList = CallbackList(),
):
    if is_train:
        # callback: start train_loop
        callbacks.on_train_start(
            logs={"num_epochs": num_epochs, "initial_cumulative_loss": cumulative_loss}
        )

        for epoch in range(num_epochs):
            # callback: start epoch
            callbacks.on_epoch_start(
                epoch=epoch, logs={"epoch": epoch, "cumulative_loss": cumulative_loss}
            )

            epoch_loss: float = 0.0  # resetting loss for each epoch
            epoch_loss = _batch_loop(
                dataloader=dataloader,
                device=device,
                total_loss=epoch_loss,
                optimizer=optimizer,
                model=model,
                criterion=criterion,
                callbacks=callbacks,
            )
            cumulative_loss += epoch_loss

            # callback: end epoch
            callbacks.on_epoch_end(
                epoch=epoch,
                logs={
                    "epoch": epoch,
                    "epoch_loss": epoch_loss,
                    "cumulative_loss": cumulative_loss,
                    "model_state": model.state_dict(),
                },
            )

        # callback: end train_loop
        callbacks.on_train_end(
            logs={
                "final_cumulative_loss": cumulative_loss,
                "final_model_state": model.state_dict(),
            }
        )
    else:
        model.eval()

        # callback: start validation
        callbacks.on_validation_start(logs={"initial_cumulative_loss": cumulative_loss})

        with torch.no_grad():
            loss: float = cumulative_loss
            loss = _batch_loop(
                dataloader=dataloader,
                device=device,
                total_loss=loss,
                model=model,
                criterion=criterion,
                callbacks=callbacks,
            )

        # callback: end validation
        callbacks.on_validation_end(
            logs={"validation_loss": loss, "model_state": model.state_dict()}
        )


def test_loop(
    testloader: DataLoader,
    model: Model,
    device: Device,
    callbacks: CallbackList = CallbackList(),
):
    model.eval()

    # callback: start testing
    callbacks.on_test_start(
        logs={
            "test_loader": testloader,
            "model_state": model.state_dict(),
            "device": device,
        }
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader):
            # callback: start test_batch
            callbacks.on_batch_start(
                batch=batch_idx, logs={"batch_idx": batch_idx, "batch_data": batch}
            )

            inputs: Tensor = batch[0].to(device)
            labels: Tensor = batch[1].to(device)
            predictions: Tensor = model(inputs)

            # callback: end test_batch
            callbacks.on_batch_end(
                batch=batch_idx,
                logs={
                    "batch_idx": batch_idx,
                    "inputs": inputs,
                    "labels": labels,
                    "predictions": predictions,
                },
            )

    # callback: end testing
    callbacks.on_test_end(
        logs={"final_model_state": model.state_dict(), "num_batches": len(testloader)}
    )
