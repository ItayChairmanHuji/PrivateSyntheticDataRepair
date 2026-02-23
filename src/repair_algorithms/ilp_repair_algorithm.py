import gurobipy as gp

from src.datasets.dataset import Dataset
from src.loggers.logger import Logger
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.repair_algorithms.repair_algorithm import RepairAlgorithm


class ILPRepairAlgorithm(RepairAlgorithm):
    def __init__(self, gp_license: dict, consider_marginals: bool, logger: Logger):
        super().__init__(logger)
        self.consider_marginals = consider_marginals
        self.env = self._configure_env(gp_license)

    @staticmethod
    def _configure_env(gp_license: dict) -> gp.Env:
        env = gp.Env(params=gp_license)
        env.setParam("OutputFlag", True)
        env.setParam("MIPGap", 0)
        env.setParam("MIPGapAbs", 0)
        env.setParam("TimeLimit", 7200)
        return env

    def _get_indices_to_remove(self, data: Dataset, mcs: MarginalsConstraints) -> list[int]:
        model = self._build_model(data, mcs)
        self.timer(lambda: model.optimize())
        return self._analyze_solution(model)

    def _build_model(self, data: Dataset, mcs: MarginalsConstraints) -> gp.Model:
        model = gp.Model("ILP_Repair", env=self.env)
        objective = model.addVars(range(len(data)), vtype=gp.GRB.BINARY)
        self._no_trivial_solution_constraint(model, objective)
        self._no_violations_constraints(model, objective, data)
        if self.consider_marginals:
            self._marginals_constraints(model, objective, data, mcs)
        model.setObjective(objective.sum(), gp.GRB.MAXIMIZE)
        return model

    @staticmethod
    def _analyze_solution(model: gp.Model) -> list[int]:
        objective = model.getVars()
        if model.status == gp.GRB.INFEASIBLE:
            return list(range(len(objective)))
        return [i for i in range(len(objective)) if objective[i].X < 0.5]

    @staticmethod
    def _no_trivial_solution_constraint(model, objective):
        model.addConstr(objective.sum() >= 1)

    @staticmethod
    def _no_violations_constraints(model, objective, data):
        for pair in data.find_violations().values:
            model.addConstr(objective[pair[0]] + objective[pair[1]] <= 1)

    @staticmethod
    def _marginals_constraints(model, objective, data, mcs):
        for indices, threshold in zip(mcs.indices(data), mcs.thresholds):
            if not indices:
                if threshold > 0:
                    model.addConstr(objective.sum() == 0)
                continue
            model.addConstr(
                gp.quicksum(objective[i] for i in indices)
                >= threshold * objective.sum()
            )
