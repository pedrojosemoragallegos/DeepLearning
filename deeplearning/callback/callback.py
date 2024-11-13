class Callback:
    def on_train_batch_start(self, **kwargs) -> None:
        pass

    def on_train_batch_end(self, **kwargs) -> None:
        pass

    def on_val_batch_start(self, **kwargs) -> None:
        pass

    def on_val_batch_end(self, **kwargs) -> None:
        pass

    def on_epoch_start(self, **kwargs) -> None:
        pass

    def on_epoch_end(self, **kwargs) -> None:
        pass

    def on_train_start(self, **kwargs) -> None:
        pass

    def on_train_end(self, **kwargs) -> None:
        pass

    def on_validation_start(self, **kwargs) -> None:
        pass

    def on_validation_end(self, **kwargs) -> None:
        pass

    def on_test_start(self, **kwargs) -> None:
        pass

    def on_test_end(self, **kwargs) -> None:
        pass

    def on_training_stopped(self, **kwargs) -> None:
        pass

    def on_checkpoint_save(self, **kwargs) -> None:
        pass

    def on_checkpoint_load(self, **kwargs) -> None:
        pass

    def on_test_batch_start(self, **kwargs) -> None:
        pass

    def on_test_batch_end(self, **kwargs) -> None:
        pass
