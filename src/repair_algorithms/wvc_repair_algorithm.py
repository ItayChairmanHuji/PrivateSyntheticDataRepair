from typing import Callable, TypeAlias

import numpy as np
from igraph import Graph, Vertex

from src.datasets.dataset import Dataset
from src.loggers.logger import Logger
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.repair_algorithms.repair_algorithm import RepairAlgorithm
from src.repair_algorithms.weight_functions.weight_function import WeightFunction

WeightFunctionTemplate: TypeAlias = Callable[[Dataset, MarginalsConstraints], WeightFunction]


class VertexCoverRepairAlgorithm(RepairAlgorithm):
    def __init__(self, weight_function_template: WeightFunctionTemplate, logger: Logger):
        super().__init__(logger)
        self.weight_func_template = weight_function_template

    def _get_indices_to_remove(self, data: Dataset, mcs: MarginalsConstraints) -> list[int]:
        graph = self._build_graph(data)
        weight_func = self.weight_func_template(data, mcs)
        cover = self.timer(lambda: self._find_cover(graph, weight_func))
        return cover

    @staticmethod
    def _build_graph(data: Dataset) -> Graph:
        return Graph(
            n=len(data),
            edges=data.find_violations().values,
            vertex_attrs={"tuple": list(data.index)},
        )

    def _find_cover(self, graph: Graph, weight_func: WeightFunction) -> list[int]:
        cover = []
        while graph.ecount() > 0:
            print("Remaining edges:", graph.ecount())
            vertex = self._select(graph, weight_func)
            self._remove(graph, vertex.index)
            weight_func.update(vertex["tuple"])
            cover.append(vertex["tuple"])
        return cover

    def _select(self, graph: Graph, weight_func) -> Vertex:
        vertices = graph.vs.select(_degree_gt=0)
        weights = weight_func(np.array([v["tuple"] for v in vertices]))
        degrees = self._get_degrees(vertices)
        self.logger.log("weights_variance", weights.var())
        self.logger.log("degrees_variance", degrees.var())
        self.logger.log("score_variance", (weights / degrees).var())
        return vertices[np.argmin(weights / degrees)]

    @staticmethod
    def _remove(graph: Graph, vertex_index: int) -> None:
        graph.delete_edges(graph.incident(vertex_index))

    @staticmethod
    def _get_degrees(vertices: list[Vertex]) -> np.ndarray:
        eps = 1e-10
        degrees = np.array(vertices.degree())
        normalization = np.max(degrees) - np.min(degrees) + eps
        shifted_degrees = degrees - np.min(degrees) + eps
        return shifted_degrees / normalization
