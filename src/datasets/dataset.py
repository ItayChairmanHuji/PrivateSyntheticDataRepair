import pandas as pd

from src.datasets.dcs.denial_constraints import DenialConstraints


class Dataset:
    def __init__(self, name: str, dcs: DenialConstraints, path: str = None, data: pd.DataFrame = None,
                 target: str = None, sample_size: int = None):
        self.name = name
        if path is None and data is None:
            raise ValueError("Either path or data must be provided")
        self.data = pd.read_csv(path) if data is None else data
        if sample_size is not None:
            self.data = self.data.sample(sample_size).reset_index(drop=True)
        self.dcs = dcs
        self.target = target

    def find_violations(self) -> pd.DataFrame:
        return self.dcs.find_violations(self.data)

    def count_violations(self) -> int:
        return len(self.find_violations())

    def __len__(self):
        return len(self.data)

    @property
    def index(self):
        return self.data.index

    @property
    def columns(self):
        return self.data.columns

    @property
    def features(self):
        if self.target is None:
            return self.data
        return [col for col in self.data.columns if col != self.target]
