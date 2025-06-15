from collections import defaultdict
import numpy as np
# from .symbolic import ComputationGraph # Assuming ComputationGraph will be imported if type hinting is strict

class PrimitiveCatalog:
    def __init__(self):
        # Stores op_name -> list of (sequence, cost_vector) tuples
        self._catalog = defaultdict(list)

    def register(self, op_name: str, sequence: list[str], cost_vector: list[float]):
        if not isinstance(op_name, str):
            raise TypeError("op_name must be a string.")
        if not op_name.strip():
            raise ValueError("op_name cannot be empty or just whitespace.")
        if not isinstance(sequence, list) or not all(isinstance(s, str) for s in sequence):
            raise TypeError("sequence must be a list of strings.")
        if not sequence:
            raise ValueError("sequence cannot be empty.")
        if not all(s.strip() for s in sequence): # Ensure no empty strings in sequence
            raise ValueError("Primitive names in sequence cannot be empty or just whitespace.")
        if not isinstance(cost_vector, list) or not all(isinstance(c, (int, float)) for c in cost_vector):
            raise TypeError("cost_vector must be a list of numbers (int or float).")

        self._catalog[op_name].append((list(sequence), list(cost_vector))) # Store copies

    def list_candidates(self, op_name: str) -> list[tuple[list[str], list[float]]]:
        if not isinstance(op_name, str):
            raise TypeError("op_name must be a string.")
        return self._catalog.get(op_name, [])

# Other classes like Optimizer will be added later in this file
# (as per the prompt)

class Optimizer:
    def __init__(self, catalog: 'PrimitiveCatalog', weights: list[float]):
        if not isinstance(catalog, PrimitiveCatalog): # Make sure catalog is of the correct type
            raise TypeError("catalog must be an instance of PrimitiveCatalog.")
        if not isinstance(weights, list) or not all(isinstance(w, (int, float)) for w in weights):
            raise TypeError("weights must be a list of numbers (int or float).")
        if not weights:
            raise ValueError("weights list cannot be empty.")

        self.catalog = catalog
        self.weights = np.array(weights, dtype=float)

    def _score(self, cost_vector: list[float]) -> float:
        if not isinstance(cost_vector, list) or not all(isinstance(c, (int, float)) for c in cost_vector):
            raise TypeError("cost_vector must be a list of numbers (int or float).")

        cost_vector_np = np.array(cost_vector, dtype=float)
        if self.weights.shape[0] != cost_vector_np.shape[0]:
            raise ValueError(
                f"Dimension mismatch: weights have {self.weights.shape[0]} elements, "
                f"cost_vector has {cost_vector_np.shape[0]} elements."
            )
        return np.dot(self.weights, cost_vector_np)

    def optimize(self, graph: 'ComputationGraph') -> list[tuple[str, list[float]]]:
        # Type hint for graph can be 'ComputationGraph' as a string if symbolic is not imported
        # to avoid circular dependencies, or import it if it's safe.

        # Ensure graph object is of a type that has topological_sort and _g.nodes[node_id]['name']
        if not hasattr(graph, 'topological_sort') or not hasattr(graph, '_g'):
            raise TypeError("graph object does not have required methods/attributes (topological_sort, _g).")

        # Assuming graph._g is a networkx.DiGraph and graph.topological_sort() returns list of node IDs
        # Also assuming graph nodes (graph._g.nodes[node_id]) have a 'name' attribute for the op_name.

        sorted_op_nodes = graph.topological_sort()
        if not sorted_op_nodes: # Handles if graph was empty or build_from_intent resulted in no nodes
            return []

        flat_plan = []
        for node_id in sorted_op_nodes:
            try:
                # Check if node exists and has 'name' attribute
                if node_id not in graph._g:
                     raise ValueError(f"Node ID {node_id} from topological sort not found in graph.")
                node_data = graph._g.nodes[node_id]
                if 'name' not in node_data:
                    raise ValueError(f"Node {node_id} in graph is missing 'name' attribute.")
                op_name = node_data['name']
            except KeyError:
                # This might occur if node_id from topological_sort isn't in graph._g.nodes,
                # which shouldn't happen if topological_sort is derived from graph._g.
                raise ValueError(f"Node {node_id} (from topological sort) data could not be accessed in graph.")


            candidates = self.catalog.list_candidates(op_name)
            if not candidates:
                raise ValueError(f"No mapping for op: {op_name} (from graph node {node_id})")

            min_score = float('inf')
            best_sequence_info = None # Stores (sequence, cost_vector) tuple

            for sequence, cost_vector in candidates:
                try:
                    current_score = self._score(cost_vector)
                except ValueError as e: # Catch dimension mismatch from _score
                    raise ValueError(
                        f"Error scoring candidate for op {op_name} (sequence {sequence}): {e}"
                    )

                if current_score < min_score:
                    min_score = current_score
                    best_sequence_info = (sequence, cost_vector)

            if best_sequence_info:
                chosen_sequence, chosen_cost_vector = best_sequence_info
                # The plan is a flat list of primitive names, each associated with the *original op's* chosen cost vector
                for primitive_name in chosen_sequence:
                    flat_plan.append((primitive_name, list(chosen_cost_vector)))
            else:
                # This case implies candidates list was empty, which is checked above.
                # Or, all candidates had non-numeric scores (inf), which _score should prevent if inputs are numbers.
                # If this path is reached, it's an unexpected state, possibly due to bad candidate data.
                # For robustness, one might log or raise an error specific to this scenario.
                # However, the `if not candidates:` check should cover this.
                pass

        return flat_plan

    def optimize_batch(self, graph: 'ComputationGraph', weight_vectors: list[list[float]]):
        # TODO: Implement batch optimization if needed for meta-learning
        # This would likely iterate through weight_vectors, call optimize (or a modified version)
        # for each, and collect the results.
        raise NotImplementedError("Batch optimization is not yet implemented.")
```
