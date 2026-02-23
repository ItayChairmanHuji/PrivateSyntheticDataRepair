from abc import ABC

from src.datasets.dataset import Dataset
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class QualityMetric(ABC):
    def __call__(self, private_data: Dataset, synthetic_data: Dataset, repaired_data: Dataset,
                 mcs: MarginalsConstraints) -> dict[str, float]:
        raise NotImplementedError("Subclasses must implement this method")
