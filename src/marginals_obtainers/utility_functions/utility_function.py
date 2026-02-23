from abc import ABC, abstractmethod

import numpy as np


class UtilityFunction(ABC):
    @abstractmethod
    def __call__(self, private_marginal: np.ndarray, synthetic_marginal: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute method")
