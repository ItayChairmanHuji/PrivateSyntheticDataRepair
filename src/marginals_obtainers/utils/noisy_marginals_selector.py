import itertools

import numpy as np
from scipy.stats import gumbel_r

from src.datasets.dataset import Dataset
from src.marginals_obtainers.utility_functions.utility_function import UtilityFunction


class NoisyMarginalsSelector:
    def __init__(self, utility_function: UtilityFunction, privacy_budget: float, is_downstream: bool = False,
                 attrs: list = None):
        self.utility_func = utility_function
        self.privacy_budget = privacy_budget
        self._marginal_keys: list[tuple[str, str, str, str]] = []
        self._scores: np.ndarray = np.zeros(0)
        self.is_downstream = is_downstream
        self.attrs = [] if attrs is None else attrs

    def fit(self, private_data: Dataset, synthetic_data: Dataset) -> "NoisyMarginalsSelector":
        priv_marginals = self._compute_marginals(private_data)
        syn_marginals = self._compute_marginals(synthetic_data)
        self._compute_utility_scores(priv_marginals, syn_marginals)
        return self

    def select(self, num_marginals: int) -> list[tuple[str, str, str, str]]:
        noise = self._get_noise(num_marginals)
        noisy_scores = self._scores + noise
        sorted_indices = np.argsort(noisy_scores)[::-1]
        top_k_indices = sorted_indices[:num_marginals]
        return [self._marginal_keys[i] for i in top_k_indices]

    def _compute_marginals(self, data: Dataset) -> dict[tuple[str, str, str, str], float]:
        result: dict[tuple[str, str, str, str], float] = {}

        attrs = self.attrs if len(self.attrs) > 0 else itertools.combinations(data.columns, 2)
        for attr1, attr2 in attrs:
            if attr1 != data.target and attr2 != data.target and self.is_downstream:
                continue

            pair_counts = (
                data.data[[attr1, attr2]]
                .value_counts(dropna=False, normalize=True)
                .items()
            )

            for (val1, val2), freq in pair_counts:
                result[(attr1, attr2, str(val1), str(val2))] = freq

        return result

    def _compute_utility_scores(self, priv_marginals: dict, syn_marginals: dict) -> None:
        self._marginal_keys = list(priv_marginals.keys())
        priv_marg = np.array([priv_marginals[key]
                              for key in self._marginal_keys])
        syn_marg = np.array([syn_marginals.get(key, 0.0)
                             for key in self._marginal_keys])
        self._scores = self.utility_func(priv_marg, syn_marg)

    def _get_noise(self, size: int) -> np.ndarray:
        if self.privacy_budget is None:
            return np.zeros(len(self._marginal_keys))
        gumbel_scale = 2 * np.sqrt(size / 8 * self.privacy_budget / 4)
        return gumbel_r.rvs(loc=0, scale=gumbel_scale, size=len(self._marginal_keys))
