from virtual_layer.symbolic import ComputationGraph # For type hinting

# Custom Exception
class OptimizationError(Exception):
    """Custom exception for errors during the optimization process."""
    pass

class PrimitiveCatalog:
    def __init__(self):
        self.primitives = {} # Key: op_name (str), Value: list of primitive dicts

    def register(self, op_name: str, primitive_id: str, steps: list[str], cost_vector: dict):
        """
        Registers a new primitive for a given operation type.

        Args:
            op_name (str): The name of the operation (e.g., 'ADD', 'LOAD').
            primitive_id (str): A unique identifier for this primitive implementation.
            steps (list[str]): A list of human-readable steps or sub-operations.
            cost_vector (dict): A dictionary where keys are cost dimensions (e.g., 'cpu', 'memory')
                                and values are the costs.
        """
        if op_name not in self.primitives:
            self.primitives[op_name] = []

        # Check for duplicate primitive_id for the same op_name (optional, but good practice)
        for p in self.primitives[op_name]:
            if p['id'] == primitive_id:
                # Or update, or raise error, depending on desired behavior
                # For now, let's assume updates are allowed by re-registering
                p['steps'] = steps
                p['cost_vector'] = cost_vector
                return

        self.primitives[op_name].append({
            'id': primitive_id,
            'steps': steps,
            'cost_vector': cost_vector
        })

    def list_candidates(self, op_name: str) -> list[dict]:
        """
        Returns a list of primitive dictionaries for the given op_name.
        Returns an empty list if no primitives are registered for that op_name.
        """
        return self.primitives.get(op_name, [])


class Optimizer:
    def __init__(self, catalog: PrimitiveCatalog):
        self.catalog = catalog

    def _score_primitive(self, primitive: dict, cost_weights: dict) -> float:
        """
        Calculates the weighted sum of costs for a given primitive.

        Args:
            primitive (dict): The primitive dictionary {'id': ..., 'steps': ..., 'cost_vector': ...}.
            cost_weights (dict): A dictionary where keys are cost dimensions and values are their weights.

        Returns:
            float: The calculated score.
        """
        score = 0.0
        primitive_costs = primitive.get('cost_vector', {})

        for cost_dim, cost_val in primitive_costs.items():
            weight = cost_weights.get(cost_dim, 0) # Assume weight 0 if not specified in cost_weights
            score += cost_val * weight

        # Optionally, handle costs in cost_weights not present in primitive_costs
        # (e.g., if a weight exists for 'gpu' but primitive has no 'gpu' cost)
        # For now, they don't contribute to score if not in primitive's cost_vector.

        return score

    def optimize(self, graph: ComputationGraph, cost_weights: dict) -> list[tuple[dict, float]]:
        """
        Optimizes the computation graph by selecting the best primitive for each operation
        based on the provided cost weights.

        Args:
            graph (ComputationGraph): The graph to optimize.
            cost_weights (dict): Weights for different cost dimensions.

        Returns:
            list[tuple[dict, float]]: A list of tuples, where each tuple contains the
                                      selected primitive dictionary and its score.
                                      This is the PlanSequence.

        Raises:
            OptimizationError: If no candidates are found for an operation in the graph.
        """
        if not graph.graph: # Handle empty graph
            return []

        try:
            # Each item in sorted_op_ids is an op_id (str)
            sorted_op_ids = graph.topological_sort()
        except ValueError as e: # Raised by topological_sort if cycle
            raise OptimizationError(f"Cannot optimize graph with cycle: {e}")

        plan_sequence = []

        for op_id in sorted_op_ids:
            node_attrs = graph.graph.nodes[op_id]
            op_type = node_attrs.get('type')
            # op_args = node_attrs.get('args', []) # Not directly used in selection yet

            if not op_type:
                raise OptimizationError(f"Operation node {op_id} is missing 'type' attribute.")

            candidates = self.catalog.list_candidates(op_type)
            if not candidates:
                raise OptimizationError(f"No primitive candidates found for operation type '{op_type}' (node {op_id}).")

            best_primitive = None
            min_score = float('inf')

            for candidate_primitive in candidates:
                score = self._score_primitive(candidate_primitive, cost_weights)
                if score < min_score:
                    min_score = score
                    best_primitive = candidate_primitive

            if best_primitive is None:
                # This should not happen if candidates list was not empty,
                # but as a safeguard:
                raise OptimizationError(f"Could not select a best primitive for '{op_type}' (node {op_id}), though candidates existed.")

            plan_sequence.append((best_primitive, min_score))

        return plan_sequence

if __name__ == '__main__':
    # Example Usage
    catalog = PrimitiveCatalog()
    catalog.register("LOAD", "load_fast", ["step_load_f1"], {'cpu': 5, 'io': 10, 'mem': 2})
    catalog.register("LOAD", "load_mem_efficient", ["step_load_m1"], {'cpu': 10, 'io': 15, 'mem': 1})
    catalog.register("PROCESS", "process_v1", ["proc_s1", "proc_s2"], {'cpu': 20, 'mem': 5})
    catalog.register("PROCESS", "process_v2_fast", ["proc_f1"], {'cpu': 10, 'mem': 8})
    catalog.register("SAVE", "save_std", ["save_s1"], {'cpu': 5, 'io': 20})

    # Create a sample graph
    intent_dict = {
        'ops': [
            {'op': 'LOAD', 'args': ['data.csv']},
            {'op': 'PROCESS', 'args': ['input_data']},
            {'op': 'SAVE', 'args': ['output.dat']}
        ],
        'vars': {}
    }
    comp_graph = ComputationGraph()
    comp_graph.build_from_intent(intent_dict)

    optimizer = Optimizer(catalog)

    # Scenario 1: CPU is most important
    weights_cpu_focused = {'cpu': 0.7, 'io': 0.2, 'mem': 0.1}
    print(f"\n--- Optimizing with CPU-focused weights: {weights_cpu_focused} ---")
    try:
        plan_cpu = optimizer.optimize(comp_graph, weights_cpu_focused)
        for primitive, score in plan_cpu:
            print(f"Selected: {primitive['id']} (for op type, derived from primitive content), Score: {score:.2f}, Steps: {primitive['steps']}")
    except OptimizationError as e:
        print(f"Optimization Error: {e}")

    # Scenario 2: Memory is most important
    weights_mem_focused = {'cpu': 0.1, 'io': 0.1, 'mem': 0.8}
    print(f"\n--- Optimizing with Memory-focused weights: {weights_mem_focused} ---")
    try:
        plan_mem = optimizer.optimize(comp_graph, weights_mem_focused)
        for primitive, score in plan_mem:
            print(f"Selected: {primitive['id']}, Score: {score:.2f}, Steps: {primitive['steps']}")
    except OptimizationError as e:
        print(f"Optimization Error: {e}")

    # Scenario 3: Operation with no primitives
    intent_missing_op = {
        'ops': [
            {'op': 'LOAD', 'args': ['data.csv']},
            {'op': 'UNKNOWN_OP', 'args': []}
        ],
        'vars': {}
    }
    comp_graph_missing = ComputationGraph()
    comp_graph_missing.build_from_intent(intent_missing_op)
    print(f"\n--- Optimizing with an unknown operation type ---")
    try:
        plan_missing = optimizer.optimize(comp_graph_missing, weights_cpu_focused)
    except OptimizationError as e:
        print(f"Caught expected Optimization Error: {e}")

    # Scenario 4: Empty graph
    comp_graph_empty = ComputationGraph()
    print(f"\n--- Optimizing with an empty graph ---")
    plan_empty = optimizer.optimize(comp_graph_empty, weights_cpu_focused)
    print(f"Plan for empty graph: {plan_empty}")

# The following commented-out section was illustrative and moved/removed.
# # Example of how op_type is determined for selected primitive in the print loop:
# # This is illustrative. The actual op_type comes from the graph node.
# # The primitive dict itself doesn't necessarily store which op_name it belongs to,
# # that association is via the catalog's structure.
# # When printing, one would typically iterate the graph ops and then print the chosen primitive for that op.
#
# # For the example print:
# # for idx, (primitive, score) in enumerate(plan_cpu): # Assuming plan is ordered like sorted_op_ids
# #     graph_op_id = sorted_op_ids[idx] # Need sorted_op_ids from the optimize run
# #     graph_op_type = comp_graph.graph.nodes[graph_op_id]['type']
# #     print(f"For graph op {graph_op_type} (node {graph_op_id}): Selected primitive {primitive['id']}, Score: {score:.2f}")
