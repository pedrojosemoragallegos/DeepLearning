from torch.utils.tensorboard.writer import SummaryWriter
from .callback import Callback


class TensorBoardCallback(Callback):
    def __init__(self, writer: SummaryWriter):
        self._writer: SummaryWriter = writer
        self._global_step: int = 0
        self._val_global_step: int = 0

    def on_train_batch_end(self, **kwargs):
        loss = kwargs.get("loss")
        if loss is not None:
            self._writer.add_scalar("Train Loss", loss, self._global_step)
            self._global_step += 1

    def on_validation_batch_end(self, **kwargs):
        loss = kwargs.get("loss")
        if loss is not None:
            self._writer.add_scalar("Validation Loss", loss, self._val_global_step)
            self._val_global_step += 1

    def on_train_epoch_end(self, **kwargs):
        epoch_num = kwargs.get("epoch_num")
        train_epoch_loss = kwargs.get("train_epoch_loss")
        validation_loss = kwargs.get("validation_loss")

        if train_epoch_loss is not None:
            self._writer.add_scalar("Epoch Loss/Train", train_epoch_loss, epoch_num)
        if validation_loss is not None:
            self._writer.add_scalar("Epoch Loss/Validation", validation_loss, epoch_num)

    def on_train_end(self, **kwargs):
        self._writer.close()

    def on_train_interrupt(self, **kwargs):
        self._writer.close()
