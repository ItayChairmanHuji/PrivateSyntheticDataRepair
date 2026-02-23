import numpy as np

from src.datasets.dataset import Dataset
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.repair_algorithms.weight_functions.weight_function import WeightFunction


class DynamicWeightFunction(WeightFunction):
    def __init__(self, data: Dataset, mcs: MarginalsConstraints):
        self.size = len(data)
        self.count = np.array(mcs.count_occurrences(data))
        self.thresholds = np.array(mcs.thresholds)
        self._relations: dict[int, list[int]] = {}
        self._init_relations(data, mcs)

    def _init_relations(self, data: Dataset, mcs: MarginalsConstraints):
        for mc_index, indices in enumerate(mcs.indices(data.data)):
            for index in indices:
                if index not in self._relations:
                    self._relations[index] = []
                self._relations[index].append(mc_index)

    def relations(self, index: int) -> np.ndarray:
        return np.bincount(self._relations.get(index, []), minlength=self.count.size)

    def __call__(self, tuples: np.ndarray) -> np.ndarray:
        weights = np.array([self._calc_for_tuple(t) for t in tuples])
        return self._rank_weights(weights)

    def update(self, index: int) -> None:
        relations = self.relations(index)
        self.count -= relations
        self.size -= 1

    def _calc_for_tuple(self, tuple_index: int) -> np.ndarray:
        relations = self.relations(tuple_index)
        diff = (self.count - relations).astype(float)
        ratio = np.divide(diff, self.thresholds, out=np.zeros_like(diff), where=self.thresholds != 0)
        ratio[self.thresholds == 0] = np.where(diff[self.thresholds == 0] > 0, self.size - 1, 0)
        k = np.ceil(self.size - 1 - ratio)
        k = np.clip(k, a_min=0, a_max=self.size - 1)
        return np.sum(k)

    @staticmethod
    def _rank_weights(weights: np.ndarray) -> np.ndarray:
        eps = 1e-10
        normalization = np.max(weights) - np.min(weights) + eps
        shifted_weights = weights - np.min(weights) + eps
        return shifted_weights / normalization
