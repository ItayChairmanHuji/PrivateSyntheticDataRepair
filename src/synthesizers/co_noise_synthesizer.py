import random
from typing import Any

import pandas as pd
from pandas import DataFrame

from src.datasets.dataset import Dataset
from src.datasets.dcs.dc_parser import Predicate
from src.loggers.logger import Logger
from src.synthesizers.synthesizer import Synthesizer


class CoNoiseSynthesizer(Synthesizer):
    OPERATIONS = {
        "=": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
        "<=": lambda x, y: x <= y,
        ">=": lambda x, y: x >= y,
        "<": lambda x, y: x < y,
        ">": lambda x, y: x > y,
    }

    def __init__(self, num_of_iterations: int, logger: Logger):
        super().__init__(logger)
        self.num_of_iterations = num_of_iterations

    def _get_synthetic_data(self, private_data: Dataset) -> pd.DataFrame:
        self.logger.log("synthesizer_num_of_iterations", self.num_of_iterations)
        synthetic_data = private_data.data.copy()
        for _ in range(self.num_of_iterations):
            self._iteration(synthetic_data, private_data.dcs.dcs)
        return synthetic_data

    def _iteration(self, data: DataFrame, constraints: list) -> None:
        dc = random.choice(constraints)
        sampled = data.sample(n=min(2, len(data))).reset_index()
        for predicate in dc.predicates:
            if not self._is_predicate_satisfied(sampled, predicate):
                self._add_noise_to_data(data, sampled, predicate)

    def _is_predicate_satisfied(self, tuples: DataFrame, pred: Predicate) -> bool:
        operation = self.OPERATIONS[pred.op]
        first_value = tuples.iloc[0][pred.attr1]
        second_value = tuples.iloc[1][pred.attr2] if not pred.is_value else tuples[pred.attr1].dtype.type(pred.attr2)
        return operation(first_value, second_value)

    def _add_noise_to_data(self, data: DataFrame, tuples: DataFrame, pred: Predicate) -> None:
        changed_tuple_idx = random.randint(0, 1) if not pred.is_value else pred.tuple1 - 1
        new_value, changed_attr = self._get_new_value(data, tuples, pred, changed_tuple_idx)
        original_index = tuples.iloc[changed_tuple_idx]["index"]
        data.at[original_index, changed_attr] = new_value

    def _get_new_value(self, data: DataFrame, tuples: DataFrame, pred: Predicate, changed: int) -> tuple[Any, str]:
        if pred.is_value:
            return pred.attr2, pred.attr1

        attrs = [pred.attr1, pred.attr2 if not pred.is_value else pred.attr1]
        values = [tuples.iloc[0][pred.attr1], tuples.iloc[1][pred.attr2]]
        if self._is_equality_allowed(pred):
            return values[1 - changed], attrs[changed]

        domain = data[attrs[changed]].unique().tolist()
        operation = self.OPERATIONS[pred.op]

        def valid(v):
            args = list(values)
            args[changed] = v
            return operation(*args)

        potential_values = [v for v in domain if valid(v)]
        return random.choice(potential_values) if potential_values else values[changed], attrs[changed]

    @staticmethod
    def _is_equality_allowed(predicate: Predicate) -> bool:
        return predicate.op in ["=", "<=", ">="]
