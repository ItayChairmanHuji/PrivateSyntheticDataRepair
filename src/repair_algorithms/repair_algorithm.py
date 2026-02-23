import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from src.datasets.dataset import Dataset
from src.loggers.logger import Logger
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints


class RepairAlgorithm(ABC):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.runtime = 0

    def repair(self, data: Dataset, mcs: MarginalsConstraints) -> Dataset:
        indices_to_remove = self._get_indices_to_remove(data, mcs)
        self.logger.log("runtime/repaired_data", self.runtime)
        repaired_data = data.data.drop(index=indices_to_remove).reset_index(drop=True)
        return Dataset(name=data.name, data=repaired_data, dcs=data.dcs, target=data.target)

    @abstractmethod
    def _get_indices_to_remove(self, data: Dataset, mcs: MarginalsConstraints):
        raise NotImplementedError("This method should be overridden by subclasses")

    def timer(self, func: Callable[[], Any], is_runtime: bool = True) -> Any:
        tick = time.process_time()
        result = func()
        tock = time.process_time()
        if is_runtime:
            self.runtime = tock - tick
            return result
        return result, tock - tick
