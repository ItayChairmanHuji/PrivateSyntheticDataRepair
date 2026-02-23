from src.datasets.dataset import Dataset
from src.loggers.logger import Logger
from src.marginals_obtainers.marginals_obtainer import MarginalsObtainer
from src.marginals_obtainers.mcs.marginals_constraint import MarginalConstraint
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.marginals_obtainers.utils.noisy_marginals_generator import NoisyMarginalsGenerator
from src.marginals_obtainers.utils.noisy_marginals_selector import NoisyMarginalsSelector


class TopKMarginalsObtainer(MarginalsObtainer):
    def __init__(self, num_of_marginals: int, marginals_selector: NoisyMarginalsSelector,
                 marginals_generator: NoisyMarginalsGenerator, logger: Logger):
        super().__init__(logger)
        self.num_of_marginals = num_of_marginals
        self.marginals_selector = marginals_selector
        self.marginals_generator = marginals_generator

    def _obtain_marginals(self, synthetic_data: Dataset, private_data: Dataset) -> MarginalsConstraints:
        marginals_keys, runtime = self.timer(lambda: self._select_marginals(private_data, synthetic_data),
                                             is_runtime=False)
        self.logger.log("marginals_selection_runtime", runtime)
        marginals, runtime = self.timer(lambda: self._generate_marginals(private_data, marginals_keys),
                                        is_runtime=False)
        self.logger.log("marginals_generation_runtime", runtime)
        return MarginalsConstraints(marginals)

    def _select_marginals(self, private_data: Dataset, synthetic_data: Dataset) -> list[tuple[str, str, str, str]]:
        self.logger.log("marginals_selector_privacy_budget", self.marginals_selector.privacy_budget)
        self.marginals_selector.fit(private_data, synthetic_data)
        return list(self.marginals_selector.select(self.num_of_marginals))

    def _generate_marginals(self, private_data: Dataset, marginals_keys: list[tuple[str, str, str, str]]) -> list[
        MarginalConstraint]:
        self.logger.log("marginals_generator_privacy_budget", self.marginals_generator.privacy_budget)
        relevant_pairs = list(set((key[0], key[1]) for key in marginals_keys))
        self.marginals_generator.fit(private_data, relevant_pairs)
        return [self._generate_constraint(*key) for key in marginals_keys]

    def _generate_constraint(self, attr1: str, attr2: str, value1: str, value2: str) -> MarginalConstraint:
        threshold = self.marginals_generator.get(attr1, attr2, value1, value2)
        print(f"Generated marginal constraint for ({attr1}, {attr2}, {value1}, {value2}) with threshold: {threshold}")
        return MarginalConstraint(
            attr1=attr1,
            attr2=attr2,
            value1=value1,
            value2=value2,
            threshold=threshold,
        )
