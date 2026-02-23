from dataclasses import dataclass
from typing import Any


@dataclass
class MarginalConstraint:
    attr1: str
    attr2: str
    value1: Any
    value2: Any
    threshold: float

    def indices(self, data) -> list[int]:
        return data[(data[self.attr1].astype(str) == str(self.value1))
                    & (data[self.attr2].astype(str) == str(self.value2))].index.tolist()
