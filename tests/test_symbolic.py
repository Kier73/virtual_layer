import pytest
from virtual_layer.symbolic import ComputationGraph
import os # For checking file existence in visualize test
import networkx as nx # For NetworkXUnfeasible if testing cycles, though not in this set

@pytest.fixture
def sample_intent_basic():
    return {
        "ops": [
            {"name": "LOAD", "args": ["file.csv"]},
            {"name": "PROCESS", "args": ["data", "param1"]},
            {"name": "SAVE", "args": ["output.csv"]}
        ]
    }

@pytest.fixture
def sample_intent_shorter():
    return {
        "ops": [
            {"name": "INIT", "args": []},
            {"name": "RUN", "args": ["fast"]}
        ]
    }

def test_build_graph_linear_chain(sample_intent_basic):
    graph = ComputationGraph()
    graph.build_from_intent(sample_intent_basic)

    assert graph._g.number_of_nodes() == 3
    assert graph._g.number_of_edges() == 2

    expected_node_ids = ["LOAD_0", "PROCESS_1", "SAVE_2"]
    assert list(graph._g.nodes()) == expected_node_ids # Check actual node IDs

    # Check edges
    assert graph._g.has_edge("LOAD_0", "PROCESS_1")
    assert graph._g.has_edge("PROCESS_1", "SAVE_2")

    # Check node attributes (name and args)
    assert graph._g.nodes["LOAD_0"]['name'] == "LOAD"
    assert graph._g.nodes["LOAD_0"]['args'] == ["file.csv"]
    assert graph._g.nodes["PROCESS_1"]['name'] == "PROCESS"
    assert graph._g.nodes["PROCESS_1"]['args'] == ["data", "param1"]
    assert graph._g.nodes["SAVE_2"]['name'] == "SAVE"
    assert graph._g.nodes["SAVE_2"]['args'] == ["output.csv"]

    # Check labels for visualization (optional but good to verify if set)
    assert graph._g.nodes["LOAD_0"]['label'] == "LOAD(file.csv)"
    assert graph._g.nodes["PROCESS_1"]['label'] == "PROCESS(data,param1)"
    assert graph._g.nodes["SAVE_2"]['label'] == "SAVE(output.csv)"


    sorted_nodes = graph.topological_sort()
    assert sorted_nodes == expected_node_ids

def test_build_from_intent_idempotent(sample_intent_basic, sample_intent_shorter):
    graph = ComputationGraph()

    # First build
    graph.build_from_intent(sample_intent_basic)
    assert graph._g.number_of_nodes() == 3
    assert "LOAD_0" in graph._g

    # Second build with different intent
    graph.build_from_intent(sample_intent_shorter)
    assert graph._g.number_of_nodes() == 2
    assert "INIT_0" in graph._g
    assert "LOAD_0" not in graph._g # Check that old nodes are gone
    assert graph._g.number_of_edges() == 1

    expected_node_ids_shorter = ["INIT_0", "RUN_1"]
    assert graph.topological_sort() == expected_node_ids_shorter

def test_build_empty_ops():
    graph = ComputationGraph()
    intent_empty = {"ops": []}
    graph.build_from_intent(intent_empty) # Should not raise ValueError anymore

    assert graph._g.number_of_nodes() == 0
    assert graph._g.number_of_edges() == 0
    assert graph.topological_sort() == []

def test_build_no_ops_key():
    graph = ComputationGraph()
    intent_no_ops_key = {} # No "ops" key at all
    graph.build_from_intent(intent_no_ops_key) # Should also result in an empty graph

    assert graph._g.number_of_nodes() == 0
    assert graph._g.number_of_edges() == 0
    assert graph.topological_sort() == []

def test_visualize_creates_file(tmp_path, sample_intent_basic):
    graph = ComputationGraph()
    graph.build_from_intent(sample_intent_basic)

    file_path = tmp_path / "graph.png"
    graph.visualize(str(file_path))

    assert file_path.exists()
    assert file_path.is_file()
    # Optionally, check if file size is > 0, but existence is primary for now
    # This check requires pygraphviz and graphviz to be installed and working
    # If they are not, visualize might print an error and not create the file,
    # or create an empty/small error file.
    # The `ComputationGraph.visualize` has a try-except for the draw call.
    # For this test to be robust, we'd ideally mock the pygraphviz call
    # or ensure the environment has graphviz. Assuming it might create a file
    # even if dot is not fully functional (e.g. AGraph object saves something minimal).
    # A more robust check is that visualize doesn't crash and *attempts* to make a file.
    # The current `visualize` prints error if `dot` is not found.
    # For this test, we'll assume if an error occurs, the file might not be created or be empty.
    # This test implicitly relies on Graphviz being installed.
    if file_path.exists(): # Only check size if file was created
        assert os.path.getsize(str(file_path)) > 0
    else:
        # If the file doesn't exist, it implies visualization failed silently or with a print.
        # This might be acceptable if Graphviz is not guaranteed in the test environment.
        # To make it stricter, one might check capsys for error prints from visualize.
        print(f"Warning: Visualization output file {file_path} not created. Graphviz might be missing.")


def test_topological_sort_on_empty_graph():
    graph = ComputationGraph()
    # Graph is empty, build_from_intent not called
    assert graph.topological_sort() == []

def test_visualize_on_empty_graph(tmp_path):
    graph = ComputationGraph()
    # Graph is empty, build_from_intent not called
    file_path = tmp_path / "empty_graph.png"
    graph.visualize(str(file_path))
    assert file_path.exists()
    assert file_path.is_file()
    # pygraphviz creates a valid, small PNG for empty graphs if layout/draw is called.
    if file_path.exists():
        assert os.path.getsize(str(file_path)) > 0
    else:
        print(f"Warning: Empty graph visualization output file {file_path} not created. Graphviz might be missing.")
