from typing import List, Dict, Tuple, Any
import heapq

class PrimitiveCatalog:
    """Maps logical ops to candidate physical primitive sequences and costs."""
    def __init__(self):
        # op_name -> list of (sequence, cost_vector)
        self.mapping: Dict[str, List[Tuple[List[Any], List[float]]]] = {}

    def register(self, op_name: str, sequence: List[Any], cost: List[float]):
        self.mapping.setdefault(op_name, []).append((sequence, cost))

class Optimizer:
    """Heuristic search optimizer for mapping graph ops to physical plans."""
    def __init__(self, catalog: PrimitiveCatalog, weights: List[float]):
        self.catalog = catalog
        self.weights = weights

    def _score(self, cost_vector: List[float]) -> float:
        """Compute weighted scalar score from multi-dimensional cost vector."""
        return sum(w * c for w, c in zip(self.weights, cost_vector))

    def optimize(self, graph) -> List[Any]:
        """
        Input: ComputationGraph
        Output: ordered sequence of physical actions P_opt

        Implements a simple A*-like search over op mappings using summed cost.
        Assumes graph.nodes is topologically sorted list of op_names.
        """
        plan = []
        total_cost = 0.0

        # Example: linear sequence through graph
        for op_name in graph.nodes:
            candidates = self.catalog.mapping.get(op_name, [])
            if not candidates:
                raise ValueError(f"No primitive mapping for op '{op_name}'")

            # Choose candidate with minimal weighted cost
            best_seq, best_cost = min(candidates, key=lambda x: self._score(x[1]))
            plan.extend(best_seq)
            total_cost += self._score(best_cost)

        return plan
