import time
from abc import abstractmethod, ABC
from typing import Any, Callable

from pandas import DataFrame

from src.datasets.dataset import Dataset
from src.loggers.logger import Logger


class Synthesizer(ABC):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.runtime = 0

    def synthesize(self, data: Dataset) -> Dataset:
        synth_data = self._get_synthetic_data(data)
        self.logger.log("runtime/synthetic_data", self.runtime)
        return Dataset(
            name=data.name,
            data=synth_data,
            dcs=data.dcs,
            target=data.target,
        )

    @abstractmethod
    def _get_synthetic_data(self, private_data: Dataset) -> DataFrame:
        raise NotImplementedError("This method should be overridden by subclasses")

    def timer(self, func: Callable[[], Any], is_runtime: bool = True) -> Any:
        tick = time.process_time()
        result = func()
        tock = time.process_time()
        if is_runtime:
            self.runtime = tock - tick
            return result
        return result, tock - tick
