from typing import List, Optional, Callable, Any
from abc import abstractmethod

from .callback import Callback


class CallbackList:
    @abstractmethod
    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks: List[Callback] = callbacks if callbacks else []

    @abstractmethod
    def __getattr__(self, name: str) -> Callable[..., None]:
        def wrapper(**kwargs: Any) -> None:
            for callback in self.callbacks:
                method: Optional[Callable[..., None]] = getattr(callback, name, None)
                if method:
                    method(**kwargs)

        return wrapper
