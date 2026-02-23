from src.datasets.dataset import Dataset
from src.evaluators.quality_metric import QualityMetric
from src.loggers.logger import Logger
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class Evaluator:
    def __init__(self, logger: Logger, quality_metrics: dict[str, QualityMetric], dataset_name: str):
        self.logger = logger
        self.quality_metrics = quality_metrics
        self.dataset_name = dataset_name

    def evaluate(self, private_data: Dataset, synthetic_data: Dataset, repaired_data: Dataset,
                 mcs: MarginalsConstraints) -> None:
        for name, metric in self.quality_metrics.items():
            print(f"Running quality metric: {name}")
            values = metric(private_data, synthetic_data, repaired_data, mcs)
            for sub_name, value in values.items():
                full_name = f"{name}/{sub_name}" if sub_name != "" else f"{name}"
                self.logger.log(full_name, value)
            print(f"Finished running quality metric: {name}")
