from pathlib import Path

import duckdb
from pandas import DataFrame

from src.datasets.dcs.denial_constraint import DenialConstraint


class DenialConstraints:
    BASE_QUERY_PATH = Path("src/datasets/dcs/query.sql")

    def __init__(self, dcs: list[DenialConstraint]):
        self.dcs = dcs

    def find_violations(self, data: DataFrame) -> DataFrame:
        query = self._build_query()
        with duckdb.connect() as connection:
            connection.register("data", data)
            violations = connection.execute(query).df()
        return violations

    def _build_query(self) -> str:
        base_query = self._load_base_query()
        join_condition = " OR ".join(dc.as_sql for dc in self.dcs)
        return base_query.replace("<REPLACE_ME>", join_condition)

    def _load_base_query(self) -> str:
        return self.BASE_QUERY_PATH.read_text(encoding="utf-8")
