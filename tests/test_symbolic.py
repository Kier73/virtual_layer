import pytest
import networkx as nx
import os
from virtual_layer.symbolic import ComputationGraph

# Helper function to create sample intent dictionaries
def create_sample_intent(ops_list=None):
    if ops_list is None:
        ops_list = []
    return {'ops': ops_list, 'vars': {}}

@pytest.fixture
def empty_graph():
    """Provides an empty ComputationGraph instance."""
    return ComputationGraph()

@pytest.fixture
def single_op_graph():
    """Provides a ComputationGraph built from a single operation."""
    graph = ComputationGraph()
    intent = create_sample_intent([{'op': 'LOAD', 'args': ['data.csv']}])
    graph.build_from_intent(intent)
    return graph, intent

@pytest.fixture
def multi_op_graph():
    """Provides a ComputationGraph built from multiple operations."""
    graph = ComputationGraph()
    intent = create_sample_intent([
        {'op': 'LOAD', 'args': ['file.csv', 'df']},
        {'op': 'FILTER', 'args': ['df', 'colA > 10', 'df_filtered']},
        {'op': 'SAVE', 'args': ['df_filtered', 'output.csv']}
    ])
    graph.build_from_intent(intent)
    return graph, intent

# --- Tests for build_from_intent ---
def test_build_from_intent_empty(empty_graph):
    intent = create_sample_intent()
    empty_graph.build_from_intent(intent)
    assert empty_graph.graph.number_of_nodes() == 0
    assert empty_graph.graph.number_of_edges() == 0

def test_build_from_intent_single_op(single_op_graph):
    graph, intent = single_op_graph
    op_spec = intent['ops'][0]

    assert graph.graph.number_of_nodes() == 1
    assert graph.graph.number_of_edges() == 0

    # Node ID is generated, so we get it from the graph
    node_id = list(graph.graph.nodes())[0]
    node_attrs = graph.graph.nodes[node_id]

    assert node_attrs['type'] == op_spec['op']
    assert node_attrs['args'] == op_spec['args']
    assert node_attrs['label'] == f"{op_spec['op']}({', '.join(map(str, op_spec['args']))})"

def test_build_from_intent_multiple_ops(multi_op_graph):
    graph, intent = multi_op_graph
    ops = intent['ops']
    num_ops = len(ops)

    assert graph.graph.number_of_nodes() == num_ops
    assert graph.graph.number_of_edges() == num_ops - 1 # Linear chain

    # Get nodes in order of addition (which should be topological for linear graph)
    # This relies on internal node ID generation, better to use topological sort for verification
    # For now, we assume node IDs like "op_0_LOAD", "op_1_FILTER", "op_2_SAVE"

    # Check node attributes and edge connections
    # This is a bit fragile if node ID generation changes significantly
    expected_node_ids = [f"op_{i}_{op['op']}" for i, op in enumerate(ops)]

    for i, op_spec in enumerate(ops):
        node_id = expected_node_ids[i]
        assert node_id in graph.graph
        node_attrs = graph.graph.nodes[node_id]
        assert node_attrs['type'] == op_spec['op']
        assert node_attrs['args'] == op_spec['args']

        if i > 0:
            prev_node_id = expected_node_ids[i-1]
            assert graph.graph.has_edge(prev_node_id, node_id)

# --- Tests for add_node and add_edge (implicitly tested by build_from_intent) ---
# Can add direct tests if desired, but build_from_intent covers them for now.

# --- Tests for topological_sort ---
def test_topological_sort_empty(empty_graph):
    assert empty_graph.topological_sort() == []

def test_topological_sort_single_op(single_op_graph):
    graph, _ = single_op_graph
    sorted_nodes = graph.topological_sort()
    assert len(sorted_nodes) == 1
    assert sorted_nodes[0] == "op_0_LOAD" # Based on current ID generation

def test_topological_sort_linear_graph(multi_op_graph):
    graph, intent = multi_op_graph
    sorted_nodes = graph.topological_sort()

    assert len(sorted_nodes) == len(intent['ops'])
    # Expected order based on current ID generation
    expected_ids = [f"op_{i}_{op['op']}" for i, op in enumerate(intent['ops'])]
    assert sorted_nodes == expected_ids

def test_topological_sort_cycle():
    graph_with_cycle = ComputationGraph()
    # Manually create a cycle
    n1 = graph_with_cycle.add_node("op_0_A", type="A", args=[], label="A")
    n2 = graph_with_cycle.add_node("op_1_B", type="B", args=[], label="B")
    graph_with_cycle.add_edge(n1, n2)
    graph_with_cycle.add_edge(n2, n1) # Creates a cycle

    with pytest.raises(ValueError, match="Graph has a cycle"):
        graph_with_cycle.topological_sort()

# --- Tests for visualize ---
@pytest.fixture
def dot_filepath(tmp_path):
    """Provides a temporary filepath for DOT files."""
    return os.path.join(tmp_path, "test_graph.dot")

def test_visualize_empty_graph(empty_graph, dot_filepath):
    empty_graph.visualize(dot_filepath)
    assert os.path.exists(dot_filepath)
    with open(dot_filepath, 'r') as f:
        content = f.read()
        assert "digraph EmptyGraph {}" == content.strip() # Removed \n


def test_visualize_single_op_graph(single_op_graph, dot_filepath):
    graph, _ = single_op_graph
    graph.visualize(dot_filepath)
    assert os.path.exists(dot_filepath)
    # Basic check, not validating full DOT content
    with open(dot_filepath, 'r') as f:
        content = f.read()
        assert "digraph" in content # Pydot adds a default name if not specified
        assert "op_0_LOAD" in content # Node ID

def test_visualize_multi_op_graph(multi_op_graph, dot_filepath):
    graph, _ = multi_op_graph
    graph.visualize(dot_filepath)
    assert os.path.exists(dot_filepath)
    with open(dot_filepath, 'r') as f:
        content = f.read()
        assert "digraph" in content
        assert "op_0_LOAD" in content
        assert "op_1_FILTER" in content
        assert "op_2_SAVE" in content
        assert "op_0_LOAD -> op_1_FILTER" in content # Check for an edge

# Test visualization when pydot is not available (mocking ImportError)
def test_visualize_no_pydot(mocker, single_op_graph, dot_filepath, capsys):
    graph, _ = single_op_graph

    # Mock networkx.drawing.nx_pydot.to_pydot to raise ImportError
    mocker.patch('networkx.drawing.nx_pydot.to_pydot', side_effect=ImportError("pydot not found"))

    graph.visualize(dot_filepath)

    # File should not be created by pydot
    # Depending on fallback logic (not implemented here), it might create a PNG or nothing
    # For now, we just check the warning print
    captured = capsys.readouterr()
    assert "pydot (and graphviz) not found. Cannot generate DOT file." in captured.out
    # assert not os.path.exists(dot_filepath) # If no fallback implemented

def test_visualize_graph_is_empty_after_clear(empty_graph, dot_filepath):
    # Build something, then clear
    intent = create_sample_intent([{'op': 'LOAD', 'args': ['data.csv']}])
    empty_graph.build_from_intent(intent)
    assert empty_graph.graph.number_of_nodes() == 1

    empty_graph.build_from_intent(create_sample_intent()) # Build with empty, effectively clearing
    assert empty_graph.graph.number_of_nodes() == 0

    empty_graph.visualize(dot_filepath)
    assert os.path.exists(dot_filepath)
    with open(dot_filepath, 'r') as f:
        content = f.read()
        assert "digraph EmptyGraph {}" == content.strip() # Removed \n
