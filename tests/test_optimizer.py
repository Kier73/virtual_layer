import pytest
import numpy as np
from virtual_layer.optimizer import PrimitiveCatalog, Optimizer
# ComputationGraph is not directly instantiated, only mocked.

class TestPrimitiveCatalog:
    def test_register_and_list_candidates_single_op(self):
        catalog = PrimitiveCatalog()
        seq1 = ["P1", "P2"]
        cost1 = [1.0, 2.0]
        catalog.register("OP_A", seq1, cost1)

        candidates = catalog.list_candidates("OP_A")
        assert len(candidates) == 1
        assert candidates[0][0] == seq1
        assert candidates[0][1] == cost1

        seq2 = ["P3"]
        cost2 = [0.5, 1.5]
        catalog.register("OP_A", seq2, cost2) # Register another for the same op

        candidates_updated = catalog.list_candidates("OP_A")
        assert len(candidates_updated) == 2
        # Check if both are present (order might not be guaranteed by defaultdict)
        assert (seq1, cost1) in candidates_updated
        assert (seq2, cost2) in candidates_updated

    def test_list_candidates_unknown_op(self):
        catalog = PrimitiveCatalog()
        catalog.register("OP_A", ["P1"], [1.0])
        assert catalog.list_candidates("OP_UNKNOWN") == []

    def test_register_input_validation(self):
        catalog = PrimitiveCatalog()
        with pytest.raises(TypeError, match="op_name must be a string"):
            catalog.register(123, ["s"], [1.0]) # type: ignore
        with pytest.raises(ValueError, match="op_name cannot be empty or just whitespace"):
            catalog.register(" ", ["s"], [1.0])
        with pytest.raises(TypeError, match="sequence must be a list of strings"):
            catalog.register("OP", "not_a_list", [1.0]) # type: ignore
        with pytest.raises(TypeError, match="sequence must be a list of strings"):
            catalog.register("OP", [123], [1.0]) # type: ignore
        with pytest.raises(ValueError, match="sequence cannot be empty"):
            catalog.register("OP", [], [1.0])
        with pytest.raises(ValueError, match="Primitive names in sequence cannot be empty or just whitespace"):
            catalog.register("OP", ["P1", " "], [1.0])
        with pytest.raises(TypeError, match="cost_vector must be a list of numbers"):
            catalog.register("OP", ["P1"], "not_a_list") # type: ignore
        with pytest.raises(TypeError, match="cost_vector must be a list of numbers"):
            catalog.register("OP", ["P1"], ["not_a_number"]) # type: ignore

    def test_list_candidates_input_validation(self):
        catalog = PrimitiveCatalog()
        with pytest.raises(TypeError, match="op_name must be a string"):
            catalog.list_candidates(123) # type: ignore

class MockComputationGraph:
    def __init__(self, node_op_names_map: dict):
        # node_op_names_map: e.g., {"node_id_0": "op_type_A", "node_id_1": "op_type_B"}
        # Assumes keys are already in topological order for simplicity in tests
        self._sorted_nodes_list = list(node_op_names_map.keys())
        self._g_nodes_data = {
            node_id: {'name': op_name, 'args': []} # Add dummy args
            for node_id, op_name in node_op_names_map.items()
        }

    def topological_sort(self):
        return self._sorted_nodes_list

    @property # To allow access like graph._g.nodes
    def _g(self):
        # Minimal mock for graph._g.nodes[node_id]['name']
        class MockGraphInternal:
            def __init__(self, nodes_data):
                self.nodes = nodes_data
        return MockGraphInternal(self._g_nodes_data)


class TestOptimizer:
    @pytest.fixture
    def catalog_with_primitives(self):
        catalog = PrimitiveCatalog()
        # For op1: (['P_A'], [1.0, 2.0]) and (['P_B', 'P_C'], [0.5, 3.0])
        catalog.register("op1", ["P_A"], [1.0, 2.0])
        catalog.register("op1", ["P_B", "P_C"], [0.5, 3.0])
        # For op2 (used in missing mapping test)
        catalog.register("op_other", ["P_D"], [1.0, 1.0])
        return catalog

    def test_optimizer_init_validation(self, catalog_with_primitives):
        with pytest.raises(TypeError, match="catalog must be an instance of PrimitiveCatalog"):
            Optimizer("not_a_catalog", [1.0]) # type: ignore
        with pytest.raises(TypeError, match="weights must be a list of numbers"):
            Optimizer(catalog_with_primitives, "not_a_list") # type: ignore
        with pytest.raises(TypeError, match="weights must be a list of numbers"):
            Optimizer(catalog_with_primitives, ["not_a_number"]) # type: ignore
        with pytest.raises(ValueError, match="weights list cannot be empty"):
            Optimizer(catalog_with_primitives, [])

    def test_score_method(self, catalog_with_primitives):
        optimizer = Optimizer(catalog_with_primitives, [1.0, 0.5])
        assert optimizer._score([2.0, 3.0]) == (1.0*2.0 + 0.5*3.0) # 2.0 + 1.5 = 3.5
        with pytest.raises(ValueError, match="Dimension mismatch"):
            optimizer._score([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="cost_vector must be a list of numbers"):
            optimizer._score("not_a_list") # type: ignore


    def test_optimizer_selects_best_primitive_case1(self, catalog_with_primitives):
        # Weights [1.0, 1.0]. P_A score: 1*1 + 1*2 = 3.0. P_B,P_C score: 1*0.5 + 1*3.0 = 3.5. Select P_A.
        weights = [1.0, 1.0]
        optimizer = Optimizer(catalog_with_primitives, weights)
        mock_graph = MockComputationGraph({"op1_node_0": "op1"})

        plan = optimizer.optimize(mock_graph) # type: ignore
        assert plan == [("P_A", [1.0, 2.0])]

    def test_optimizer_selects_best_primitive_case2(self, catalog_with_primitives):
        # Weights [10.0, 0.1]. P_A score: 10*1 + 0.1*2 = 10.2. P_B,P_C score: 10*0.5 + 0.1*3.0 = 5.0 + 0.3 = 5.3. Select P_B,P_C.
        weights = [10.0, 0.1]
        optimizer = Optimizer(catalog_with_primitives, weights)
        mock_graph = MockComputationGraph({"op1_node_0": "op1"})

        plan = optimizer.optimize(mock_graph) # type: ignore
        assert plan == [("P_B", [0.5, 3.0]), ("P_C", [0.5, 3.0])]

    def test_optimizer_multiple_ops_in_graph(self, catalog_with_primitives):
        catalog_with_primitives.register("opX", ["P_X"], [1.0, 1.0])
        weights = [1.0, 1.0]
        optimizer = Optimizer(catalog_with_primitives, weights)
        # op1 will choose P_A ([1.0, 2.0]), opX will choose P_X ([1.0, 1.0])
        mock_graph = MockComputationGraph({"node0": "op1", "node1": "opX"})
        plan = optimizer.optimize(mock_graph) # type: ignore
        assert plan == [("P_A", [1.0, 2.0]), ("P_X", [1.0, 1.0])]


    def test_optimizer_empty_graph(self, catalog_with_primitives):
        optimizer = Optimizer(catalog_with_primitives, [1.0, 1.0])
        mock_graph_empty = MockComputationGraph({}) # Empty node map
        assert optimizer.optimize(mock_graph_empty) == [] # type: ignore

    def test_optimizer_missing_mapping_raises_value_error(self, catalog_with_primitives):
        optimizer = Optimizer(catalog_with_primitives, [1.0, 1.0])
        mock_graph_unknown_op = MockComputationGraph({"node_unknown": "op_DOES_NOT_EXIST"})

        with pytest.raises(ValueError, match="No mapping for op: op_DOES_NOT_EXIST"):
            optimizer.optimize(mock_graph_unknown_op) # type: ignore

    def test_optimizer_score_value_error_propagates(self, catalog_with_primitives):
        # Modify a cost vector to cause dimension mismatch during scoring
        # This test is a bit tricky as list_candidates returns copies.
        # A direct way is to register a malformed one.
        catalog_with_primitives.register("op_bad_cost", ["P_BAD"], [1.0, 2.0, 3.0])
        optimizer = Optimizer(catalog_with_primitives, [1.0, 1.0]) # Weights have 2 dims
        mock_graph_bad_cost_op = MockComputationGraph({"node_bad": "op_bad_cost"})

        with pytest.raises(ValueError, match="Error scoring candidate for op op_bad_cost"):
            optimizer.optimize(mock_graph_bad_cost_op) # type: ignore

    def test_not_implemented_batch_method(self, catalog_with_primitives):
        optimizer = Optimizer(catalog_with_primitives, [1.0, 1.0])
        mock_graph = MockComputationGraph({"op1_node_0": "op1"})
        with pytest.raises(NotImplementedError):
            optimizer.optimize_batch(mock_graph, [[1.0,1.0]]) # type: ignore
```
