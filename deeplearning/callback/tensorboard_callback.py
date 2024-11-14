from typing import Optional
from torch.utils.tensorboard.writer import SummaryWriter
from .callback import Callback


class TensorBoardCallback(Callback):
    def __init__(
        self,
        train_writer: SummaryWriter,
        validation_writer: Optional[SummaryWriter],
        test_writer: Optional[SummaryWriter],
    ):
        self._train_writer: SummaryWriter = train_writer
        self._validation_writer: Optional[SummaryWriter] = validation_writer
        self._test_writer: Optional[SummaryWriter] = test_writer

        self._train_step: int = 0
        self._validation_step: int = 0
        self._test_step: int = 0

    def on_train_batch_end(self, **kwargs):
        loss: int = kwargs.get("loss")  # type: ignore

        self._train_writer.add_scalar("Loss/train", loss, self._train_step)
        self._train_step += 1

    def on_validation_batch_end(self, **kwargs):
        if self._validation_writer:
            loss: int = kwargs.get("loss")  # type: ignore

            self._validation_writer.add_scalar(
                "Loss/validation", loss, self._validation_step
            )
            self._validation_step += 1

    def on_test_batch_end(self, **kwargs):
        if self._test_writer:
            loss: int = kwargs.get("test_batch_loss")  # type: ignore

            self._test_writer.add_scalar("Loss/test", loss, self._test_step)
            self._test_step += 1

    def on_train_epoch_end(self, **kwargs):
        epoch_num: int = kwargs.get("epoch_num")  # type: ignore
        train_epoch_loss: float = kwargs.get("train_epoch_loss")  # type: ignore
        validation_loss: float = kwargs.get("validation_loss")  # type: ignore

        self._train_writer.add_scalar("Epoch Loss/Train", train_epoch_loss, epoch_num)

        if self._validation_writer:
            self._validation_writer.add_scalar(
                "Epoch Loss/Validation", validation_loss, epoch_num
            )

    def on_train_end(self, **kwargs):
        self._train_writer.close()

        if self._validation_writer:
            self._validation_writer.close()

        if self._test_writer:
            self._test_writer.close()
