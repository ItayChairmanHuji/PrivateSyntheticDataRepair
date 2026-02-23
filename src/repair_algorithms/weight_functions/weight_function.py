from abc import ABC, abstractmethod

import numpy as np


class WeightFunction(ABC):
    @abstractmethod
    def __call__(self, tuples: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement __call__ method")

    @abstractmethod
    def update(self, index: int) -> None:
        raise NotImplementedError("Subclasses must implement update method")
