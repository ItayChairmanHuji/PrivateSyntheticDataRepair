from abc import ABC, abstractmethod
from typing import Any


class Logger(ABC):
    @abstractmethod
    def log(self, metric: str, value: Any):
        raise NotImplementedError("Subclasses must implement this method")
