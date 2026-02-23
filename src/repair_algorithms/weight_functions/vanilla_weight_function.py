import numpy as np


class VanillaWeightFunction:
    def __call__(self, tuples: np.ndarray) -> np.ndarray:
        return np.ones_like(tuples)

    def update(self, index: int) -> None:
        pass