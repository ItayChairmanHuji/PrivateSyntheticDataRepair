from dataclasses import asdict

import numpy as np
import pandas as pd
from pandas import DataFrame

from src.datasets.dataset import Dataset
from src.marginals_obtainers.mcs.marginals_constraint import MarginalConstraint


class MarginalsConstraints:
    def __init__(self, mcs: list[MarginalConstraint]):
        self.mcs = mcs

    @property
    def thresholds(self) -> list[float]:
        return [mc.threshold for mc in self.mcs]

    def count_occurrences(self, data: Dataset) -> list:
        return [len(mc.indices(data.data)) for mc in self.mcs]

    def indices(self, data: Dataset) -> list[list[int]]:
        return [mc.indices(data.data) for mc in self.mcs]

    def count_satisfactions(self, data: Dataset) -> int:
        if len(data) == 0:
            return 0
        counts = self.count_occurrences(data)
        return sum(count / len(data) >= mc.threshold for count, mc in zip(counts, self.mcs))

    def count_violations(self, data: Dataset) -> int:
        if len(data) == 0:
            return 0
        counts = self.count_occurrences(data)
        return sum(count / len(data) < mc.threshold for count, mc in zip(counts, self.mcs))

    def distance(self, data: Dataset):
        if len(data) == 0:
            return np.mean(self.thresholds)
        counts = self.count_occurrences(data)
        distances = [np.maximum(mc.threshold - (count / len(data)), 0) for count, mc in zip(counts, self.mcs)]
        return np.mean(distances)

    def __iter__(self):
        return iter(self.mcs)

    def __len__(self):
        return len(self.mcs)

    def save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            df = pd.DataFrame([asdict(mc) for mc in self.mcs])
            df.to_csv(f, index=False)

    @staticmethod
    def load(filepath: str) -> 'MarginalsConstraints':
        with open(filepath, 'r') as f:
            df = pd.read_csv(f)
            mcs = [MarginalConstraint(**row) for _, row in df.iterrows()]
            return MarginalsConstraints(mcs)
