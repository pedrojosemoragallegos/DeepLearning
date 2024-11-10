from typing import List, Optional, Callable, Dict, Any


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


class CallbackList:
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks: List[Callback] = callbacks if callbacks else []

    def __getattr__(self, name: str) -> Callable[..., None]:
        def wrapper(*args: Any, **kwargs: Any) -> None:
            for callback in self.callbacks:
                method: Optional[Callable[..., None]] = getattr(callback, name, None)
                if method:
                    method(*args, **kwargs)

        return wrapper
