from logging import Logger

import pandas as pd

from src.datasets.dataset import Dataset
from src.synthesizers.synthesizer import Synthesizer


class FileSynthesizer(Synthesizer):
    def __init__(self, file_path, logger: Logger):
        super().__init__(logger)
        self.file_path = file_path

    def _get_synthetic_data(self, private_data: Dataset) -> pd.DataFrame:
        return pd.read_csv(self.file_path)
