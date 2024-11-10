from typing import List, Optional, Callable, Dict, Any


class Callback:
    """
    Base class for defining callbacks that can be used during training, validation, and testing.
    Subclasses can override these methods to customize behavior at specific points in the training loop.
    """

    def on_batch_start(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the start of a batch.

        Args:
            batch (int): Index of the current batch.
            logs (Optional[Dict[str, Any]]): Additional information about the batch.
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of a batch.

        Args:
            batch (int): Index of the current batch.
            logs (Optional[Dict[str, Any]]): Additional information about the batch.
        """
        pass

    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the start of an epoch.

        Args:
            epoch (int): Index of the current epoch.
            logs (Optional[Dict[str, Any]]): Additional information about the epoch.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of an epoch.

        Args:
            epoch (int): Index of the current epoch.
            logs (Optional[Dict[str, Any]]): Additional information about the epoch.
        """
        pass

    def on_train_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the start of training.

        Args:
            logs (Optional[Dict[str, Any]]): Additional information about the training process.
        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of training.

        Args:
            logs (Optional[Dict[str, Any]]): Additional information about the training process.
        """
        pass

    def on_validation_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the start of validation.

        Args:
            logs (Optional[Dict[str, Any]]): Additional information about the validation process.
        """
        pass

    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of validation.

        Args:
            logs (Optional[Dict[str, Any]]): Additional information about the validation process.
        """
        pass

    def on_test_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the start of testing.

        Args:
            logs (Optional[Dict[str, Any]]): Additional information about the testing process.
        """
        pass

    def on_test_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """
        Called at the end of testing.

        Args:
            logs (Optional[Dict[str, Any]]): Additional information about the testing process.
        """
        pass


class CallbackList:
    """
    A container for managing multiple callbacks. It forwards method calls to all callbacks in the list.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        """
        Initializes the CallbackList.

        Args:
            callbacks (Optional[List[Callback]]): A list of Callback instances. Defaults to an empty list.
        """
        self.callbacks: List[Callback] = callbacks if callbacks else []

    def __getattr__(self, name: str) -> Callable[..., None]:
        """
        Dynamically routes method calls to all callbacks that implement the given method.

        Args:
            name (str): The name of the method to call on each callback.

        Returns:
            Callable[..., None]: A function that calls the specified method on all callbacks.
        """

        def wrapper(*args: Any, **kwargs: Any) -> None:
            for callback in self.callbacks:
                method: Optional[Callable[..., None]] = getattr(callback, name, None)
                if method:
                    method(*args, **kwargs)

        return wrapper
