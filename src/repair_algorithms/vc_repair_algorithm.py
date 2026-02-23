import numpy as np
from igraph import Graph

from src.datasets.dataset import Dataset
from src.loggers.logger import Logger
from src.marginals_obtainers.mcs.marginals_constraints import MarginalsConstraints
from src.repair_algorithms.repair_algorithm import RepairAlgorithm


class RandomVertexCoverRepairAlgorithm(RepairAlgorithm):
    def __init__(self, logger: Logger):
        super().__init__(logger)

    def _get_indices_to_remove(self, data: Dataset, mcs: MarginalsConstraints) -> Dataset:
        graph = self._build_graph(data)
        return self.timer(lambda: self._find_cover(graph))

    @staticmethod
    def _build_graph(data: Dataset) -> Graph:
        return Graph(
            n=len(data),
            edges=data.find_violations().values,
            vertex_attrs={"tuple": list(data.index)},
        )

    @staticmethod
    def _find_cover(graph: Graph) -> list[int]:
        cover = []
        while graph.ecount() > 0:
            index = np.random.randint(0, graph.ecount())
            edge = graph.es[index]
            v1, v2 = graph.vs[edge.source], graph.vs[edge.target]
            graph.delete_edges(graph.incident(v1.index) + graph.incident(v2.index))
            cover.extend([v1["tuple"], v2["tuple"]])
        return cover
