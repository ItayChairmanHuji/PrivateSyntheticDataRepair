from pathlib import Path

import pandas as pd

from loggers.logger import Logger
from src.datasets.dataset import Dataset    
from src.marginals_obtainers.mcs.marginals_constraint import MarginalConstraint
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.marginals_obtainers.marginals_obtainer import MarginalsObtainer

class FileMarginalsObtainer(MarginalsObtainer):
    def __init__(self, file_path: str, logger: Logger):
        super().__init__(logger)
        self.file_path = Path(file_path)

    def _obtain_marginals(self, synthetic_data: Dataset, private_data: Dataset) -> MarginalsConstraints:
        df = pd.read_csv(self.file_path)
        mcs = [MarginalConstraint(**row.to_dict()) for _, row in df.iterrows()]
        return MarginalsConstraints(mcs)
