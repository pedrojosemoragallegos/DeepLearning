from typing import Optional, Dict, Any


class Callback:
    def on_batch_start(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_validation_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_test_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_test_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_training_stopped(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_checkpoint_save(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_checkpoint_load(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_test_batch_start(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    def on_test_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        pass
