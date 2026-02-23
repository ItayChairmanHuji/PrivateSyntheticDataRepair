import itertools

import numpy as np

from src.datasets.dataset import Dataset
from src.evaluators.quality_metric import QualityMetric
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class TwoWayTVDQualityMetric(QualityMetric):
    def __call__(self, private_data: Dataset, synthetic_data: Dataset,
                 repaired_data: Dataset, mcs: MarginalsConstraints) -> dict[str, float]:
        return {
            "synthetic_data": self.evaluate(synthetic_data, private_data),
            "repaired_data": self.evaluate(repaired_data, private_data),
        }

    def evaluate(self, data: Dataset, reference: Dataset) -> float:
        column_pairs = list(itertools.combinations(data.columns, 2))
        return np.mean(np.array([self._pair_tvd(data, reference, col1, col2) for col1, col2 in column_pairs]))

    @staticmethod
    def _pair_tvd(data: Dataset, reference: Dataset, col1: str, col2: str) -> float:
        a = data.data.groupby([col1, col2]).size().div(len(data.data)).rename("freq")
        b = reference.data.groupby([col1, col2]).size().div(len(reference.data)).rename("freq")
        a, b = a.align(b, fill_value=0)
        return 0.5 * np.sum(np.abs(a - b))
