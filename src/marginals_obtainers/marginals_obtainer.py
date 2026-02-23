from abc import ABC, abstractmethod
import time
from typing import Any, Callable

from src.datasets.dataset import Dataset
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.loggers.logger import Logger


class MarginalsObtainer(ABC):
    def __init__(self, logger: Logger):
        self.logger = logger

    def obtain_marginals(self, synthetic_data: Dataset, private_data: Dataset) -> MarginalsConstraints:
        mcs = self._obtain_marginals(synthetic_data, private_data)
        self.logger.log("num_of_marginals", len(mcs))
        return mcs

    @abstractmethod
    def _obtain_marginals(self, synthetic_data: Dataset, private_data: Dataset) -> MarginalsConstraints:
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def timer(self, func: Callable[[], Any], is_runtime: bool = True) -> Any: 
        tick = time.process_time()
        result = func()
        tock = time.process_time()
        if is_runtime:
            self.runtime = tock - tick
            return result
        return result, tock - tick