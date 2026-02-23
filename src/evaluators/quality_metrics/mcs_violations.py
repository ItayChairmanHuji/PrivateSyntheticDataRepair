from src.datasets.dataset import Dataset
from src.evaluators.quality_metric import QualityMetric
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class MCsViolationsQualityMetric(QualityMetric):
    def __call__(self, private_data: Dataset, synthetic_data: Dataset, repaired_data: Dataset,
                 mcs: MarginalsConstraints) -> dict[str, float]:
        return {
            "private_data": mcs.count_violations(private_data) / len(mcs),
            "synthetic_data": mcs.count_violations(synthetic_data) / len(mcs),
            "repaired_data": mcs.count_violations(repaired_data) / len(mcs)
        }
