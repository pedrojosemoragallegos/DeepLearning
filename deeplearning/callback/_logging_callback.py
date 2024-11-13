from .callback import Callback


class _LoggingCallback(Callback):
    def _log_state(self, method_name: str, **kwargs) -> None:
        sorted_kwargs = {key: kwargs[key] for key in sorted(kwargs)}
        print(f"[{method_name}] State:")
        for key, value in sorted_kwargs.items():
            print(f"  {key}: {value}")

    def on_train_batch_start(self, **kwargs) -> None:
        self._log_state("on_train_batch_start", **kwargs)

    def on_train_batch_end(self, **kwargs) -> None:
        self._log_state("on_train_batch_end", **kwargs)

    def on_val_batch_start(self, **kwargs) -> None:
        self._log_state("on_val_batch_start", **kwargs)

    def on_val_batch_end(self, **kwargs) -> None:
        self._log_state("on_val_batch_end", **kwargs)

    def on_epoch_start(self, **kwargs) -> None:
        self._log_state("on_epoch_start", **kwargs)

    def on_epoch_end(self, **kwargs) -> None:
        self._log_state("on_epoch_end", **kwargs)

    def on_train_start(self, **kwargs) -> None:
        self._log_state("on_train_start", **kwargs)

    def on_train_end(self, **kwargs) -> None:
        self._log_state("on_train_end", **kwargs)

    def on_validation_start(self, **kwargs) -> None:
        self._log_state("on_validation_start", **kwargs)

    def on_validation_end(self, **kwargs) -> None:
        self._log_state("on_validation_end", **kwargs)

    def on_test_start(self, **kwargs) -> None:
        self._log_state("on_test_start", **kwargs)

    def on_test_end(self, **kwargs) -> None:
        self._log_state("on_test_end", **kwargs)

    def on_training_stopped(self, **kwargs) -> None:
        self._log_state("on_training_stopped", **kwargs)

    def on_checkpoint_save(self, **kwargs) -> None:
        self._log_state("on_checkpoint_save", **kwargs)

    def on_checkpoint_load(self, **kwargs) -> None:
        self._log_state("on_checkpoint_load", **kwargs)

    def on_test_batch_start(self, **kwargs) -> None:
        self._log_state("on_test_batch_start", **kwargs)

    def on_test_batch_end(self, **kwargs) -> None:
        self._log_state("on_test_batch_end", **kwargs)
