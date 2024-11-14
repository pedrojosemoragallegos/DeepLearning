class Callback:
    def on_train_batch_start(self, **kwargs):
        pass

    def on_train_batch_end(self, **kwargs):
        pass

    def on_validation_batch_start(self, **kwargs):
        pass

    def on_validation_batch_end(self, **kwargs):
        pass

    def on_train_epoch_start(self, **kwargs):
        pass

    def on_train_epoch_end(self, **kwargs):
        pass

    def on_train_start(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_validationidation_start(self, **kwargs):
        pass

    def on_validationidation_end(self, **kwargs):
        pass

    def on_test_start(self, **kwargs):
        pass

    def on_test_end(self, **kwargs):
        pass

    def on_training_stopped(self, **kwargs):
        pass

    def on_test_batch_start(self, **kwargs):
        pass

    def on_test_batch_end(self, **kwargs):
        pass
