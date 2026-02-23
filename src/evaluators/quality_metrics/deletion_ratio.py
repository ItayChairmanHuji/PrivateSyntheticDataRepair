from src.datasets.dataset import Dataset
from src.evaluators.quality_metric import QualityMetric
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class DeletionRatioQualityMetric(QualityMetric):
    def __call__(self, private_data: Dataset, synthetic_data: Dataset, repaired_data: Dataset,
                 mcs: MarginalsConstraints) -> dict[str, float]:
        return {
            "private_data": 1 - len(private_data) / len(private_data),
            "synthetic_data": 1 - len(synthetic_data) / len(private_data),
            "repaired_data": 1 - len(repaired_data) / len(private_data)
        }
