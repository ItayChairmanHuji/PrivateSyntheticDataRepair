from src.datasets.dataset import Dataset
from src.evaluators.quality_metric import QualityMetric
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class DCsViolationsQualityMetric(QualityMetric):
    def __call__(self, private_data: Dataset, synthetic_data: Dataset, repaired_data: Dataset,
                 mcs: MarginalsConstraints) -> dict[str, float]:
        return {
            "private_data": private_data.count_violations(),
            "synthetic_data": synthetic_data.count_violations(),
            "repaired_data": repaired_data.count_violations()
        }
