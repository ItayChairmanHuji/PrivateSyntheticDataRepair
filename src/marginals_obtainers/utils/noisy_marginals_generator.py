from typing import Optional

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.datasets.dataset import Dataset
from src.marginals_obtainers.utils.residual_planner import ResidualPlanner


class NoisyMarginalsGenerator:
    def __init__(self, privacy_budget: float):
        self.privacy_budget = privacy_budget
        self.marginals: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        self._encoders: dict[str, LabelEncoder] = {}
        self._column_names: list[str] = []
        self._domains: list[int] = []
        self._planner: Optional[ResidualPlanner] = None

    def fit(self, data: Dataset, relevant_pairs: list[tuple[str, str]]) -> "NoisyMarginalsGenerator":
        self._initialize_from_data(data)
        self._fit_residual_planner(data, relevant_pairs)
        self.marginals = self._extract_noisy_marginals()
        return self

    def get(self, attr1: str, attr2: str, value1: str, value2: str) -> float:
        return self.marginals[(attr1, attr2)][(value1, value2)]

    def _initialize_from_data(self, data: Dataset) -> None:
        self._column_names = list(data.columns)
        self._domains = [data.data[col].nunique() for col in self._column_names]
        self._encoders = self._fit_label_encoders(data)

    def _fit_label_encoders(self, data: Dataset) -> dict:
        encoders = {}
        for col in self._column_names:
            le = LabelEncoder()
            le.fit(data.data[col].astype(str))
            encoders[col] = le
        return encoders

    def _encode_data(self, data: DataFrame) -> DataFrame:
        data_encoded = data.copy()
        for col in self._column_names:
            data_encoded[col] = self._encoders[col].transform(data_encoded[col].astype(str))
        return data_encoded

    def _fit_residual_planner(self, data: Dataset, relevant_pairs: list[tuple[str, str]]) -> None:
        planner = ResidualPlanner(self._domains)
        indices = list(set([(self._column_names.index(attr1),
                             self._column_names.index(attr2)) for attr1, attr2 in relevant_pairs]))
        for i, j in indices:
            planner.input_mech((i, j))

        encoded_data = self._encode_data(data.data)
        planner.input_data(encoded_data, self._column_names)

        if self.privacy_budget is None:
            for res_mech in planner.res_dict.values():
                res_mech.input_noise_level(0.0)
            planner.measurement()
            planner.reconstruction()
        else:
            planner.selection(choice="sumvar", pcost=self.privacy_budget)
            planner.measurement()
            planner.reconstruction()

        self._planner = planner

    def _extract_noisy_marginals(self) -> dict:
        marginals = {}
        data_size = len(self._planner.data)

        for (attr_idx1, attr_idx2), mech in self._planner.mech_dict.items():
            attr_domains = [self._domains[attr_idx1], self._domains[attr_idx2]]
            if self.privacy_budget is None:
                col1 = self._column_names[attr_idx1]
                col2 = self._column_names[attr_idx2]
                bins = [np.arange(attr_domains[0] + 1), np.arange(attr_domains[1] + 1)]
                counts = np.histogram2d(self._planner.data[col1], self._planner.data[col2], bins=bins)[0]
                noisy_table = counts / data_size
            else:
                noisy_vector = mech.get_noisy_answer()
                noisy_table = noisy_vector.reshape(attr_domains)
                min_val = noisy_table.min()
                if min_val < 0:
                    noisy_table = noisy_table - min_val
                total = noisy_table.sum()
                noisy_table = noisy_table / total
            attr1_name = self._column_names[attr_idx1]
            attr2_name = self._column_names[attr_idx2]
            marginal_dict = {}

            for v1_idx in range(attr_domains[0]):
                for v2_idx in range(attr_domains[1]):
                    v1_decoded = self._encoders[attr1_name].inverse_transform([v1_idx])[0]
                    v2_decoded = self._encoders[attr2_name].inverse_transform([v2_idx])[0]
                    marginal_dict[(v1_decoded, v2_decoded)] = noisy_table[v1_idx, v2_idx]

            marginals[(attr1_name, attr2_name)] = marginal_dict

        return marginals
