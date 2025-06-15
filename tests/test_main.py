import pytest
import json
import os
import subprocess
import sys
import logging
from unittest.mock import patch, MagicMock

from virtual_layer.main import run_pipeline, main as cli_main
from virtual_layer.dsl import DSLCompiler, DSLSyntaxError
from virtual_layer.symbolic import ComputationGraph
from virtual_layer.optimizer import PrimitiveCatalog, Optimizer, OptimizationError
from virtual_layer.executor import actuate_sequence
from virtual_layer.meta import MetaCognitiveLoop
from virtual_layer.surrogate import BasicSklearnSurrogate


# --- Fixtures ---

@pytest.fixture
def temp_files(tmp_path):
    """Create temporary files for testing."""
    files = {
        "dsl_file": tmp_path / "test_script.dsl",
        "primitives_file": tmp_path / "test_primitives.json",
        "meta_state_file": tmp_path / "test_meta_state.json",
        "graph_viz_file": tmp_path / "graph.dot",
    }
    return files

@pytest.fixture
def sample_primitives_json_content():
    return {
        "LOAD": [{"id": "P_LOAD_T1", "op_type": "LOAD", "args_template": ["filepath"], "steps": ["load_t1"], "cost_vector": {"time": 10, "cpu": 1}}],
        "ADD": [{"id": "P_ADD_T1", "op_type": "ADD", "args_template": ["var1", "var2"], "steps": ["add_t1"], "cost_vector": {"time": 1, "cpu": 1}}],
        "SAVE": [{"id": "P_SAVE_T1", "op_type": "SAVE", "args_template": ["data", "filepath"], "steps": ["save_t1"], "cost_vector": {"time": 5, "cpu": 1}}]
    }

@pytest.fixture
def setup_primitives_file(temp_files, sample_primitives_json_content):
    filepath = temp_files["primitives_file"]
    with open(filepath, 'w') as f:
        json.dump(sample_primitives_json_content, f)
    return filepath

@pytest.fixture
def default_test_weights():
    return {"time": 0.7, "cpu": 0.3, "quality": 0.0, "mem": 0.0, "io": 0.0, "network": 0.0}

@pytest.fixture
def simple_dsl_compiler(sample_primitives_json_content):
    op_names = set(sample_primitives_json_content.keys())
    op_names.update(["PROCESS"])
    return DSLCompiler(catalog=op_names)

@pytest.fixture
def simple_catalog(setup_primitives_file):
    catalog = PrimitiveCatalog()
    catalog.load_from_json(setup_primitives_file)
    if not catalog.list_candidates("PROCESS"): # Ensure PROCESS op for tests if not in json
        catalog.register("PROCESS", "P_PROCESS_dummy", ["dummy process"], {"op_type":"PROCESS", "time":1, "cpu":1})
    return catalog

@pytest.fixture
def simple_meta_loop(tmp_path, default_test_weights):
    return MetaCognitiveLoop(
        initial_weights=default_test_weights,
        state_file_path=str(tmp_path / "pipeline_meta_state.json")
    )

# --- Tests for run_pipeline (direct function call) ---

def test_run_pipeline_simple_flow(simple_catalog, simple_meta_loop, simple_dsl_compiler):
    dsl_code = 'LOAD("data.csv")\nADD(var1, 10)\nSAVE(var1, "out.csv")'
    initial_state = {'var1': 0}

    results = run_pipeline(
        code=dsl_code,
        initial_state=initial_state,
        catalog=simple_catalog,
        meta_loop=simple_meta_loop,
        dsl_compiler=simple_dsl_compiler
    )

    assert "intent" in results
    assert "graph_nodes" in results
    assert "plan" in results
    assert "final_state" in results
    assert "telemetry" in results
    assert "meta_state" in results

    assert len(results['intent']['ops']) == 3
    assert len(results['graph_nodes']) == 3
    assert len(results['plan']) == 3
    assert len(results['telemetry']) == 3

    assert results['final_state']['data.csv_status'] == "loaded"
    assert results['final_state']['result_of_add'] == "computed_sum_placeholder"
    assert results['final_state']['out.csv_status'] == "saved"

    assert results['meta_state']['total_executions'] == 3

def test_run_pipeline_with_viz(simple_catalog, simple_meta_loop, simple_dsl_compiler, temp_files):
    dsl_code = 'LOAD("data.csv")'
    initial_state = {}
    viz_path = str(temp_files["graph_viz_file"])

    if os.path.exists(viz_path): os.remove(viz_path)

    run_pipeline(
        code=dsl_code,
        initial_state=initial_state,
        catalog=simple_catalog,
        meta_loop=simple_meta_loop,
        dsl_compiler=simple_dsl_compiler,
        graph_viz_path=viz_path
    )
    assert os.path.exists(viz_path)
    with open(viz_path, 'r') as f:
        content = f.read()
        assert "digraph" in content
        assert "op_0_LOAD" in content

# --- Tests for main() CLI execution ---

def run_cli(args_list, temp_dir_path):
    command = [sys.executable, "-m", "virtual_layer.main"] + args_list
    env = os.environ.copy()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env["PYTHONPATH"] = project_root + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    return subprocess.run(command, capture_output=True, text=True, check=False, env=env, cwd=str(temp_dir_path))

def test_cli_minimal_args_dsl_string(temp_files, sample_primitives_json_content, default_test_weights, tmp_path):
    dsl_string_fixed = 'LOAD("data.csv")\nSAVE("output.csv")'

    primitives_file_path = tmp_path / "prims.json"
    with open(primitives_file_path, 'w') as f:
        json.dump(sample_primitives_json_content, f)

    meta_file_path = tmp_path / "meta.json"
    if os.path.exists(meta_file_path): os.remove(meta_file_path)

    result = run_cli([
        dsl_string_fixed,
        "--primitives-file", str(primitives_file_path.name),
        "--meta-state-file", str(meta_file_path.name),
        "--initial-weights-json", json.dumps(default_test_weights),
        "--log-level", "INFO"
    ], tmp_path)

    assert result.returncode == 0, f"CLI Error: {result.stderr}\nStdout: {result.stdout}"
    assert os.path.exists(meta_file_path)
    assert "Pipeline run completed." in result.stdout
    assert "Final Meta-State" in result.stdout
    with open(meta_file_path, 'r') as f:
        meta_data = json.load(f)
        assert meta_data['total_executions'] == 2

@patch('virtual_layer.main.run_pipeline')
@patch('argparse.ArgumentParser.parse_args')
def test_cli_main_function_call_direct(mock_parse_args, mock_run_pipeline, tmp_path,
                                     sample_primitives_json_content, default_test_weights, caplog):
    caplog.set_level(logging.INFO)
    mock_args = MagicMock()
    mock_args.dsl_code_or_file = 'LOAD("x.csv")'
    mock_args.initial_state_json = "{}"
    mock_args.initial_weights_json = json.dumps(default_test_weights)
    mock_args.learning_rate = 0.05
    mock_args.target_fidelity = 0.9
    mock_args.meta_state_file = str(tmp_path / "meta_cli.json")
    mock_args.primitives_file = str(tmp_path / "prims_cli.json")
    mock_args.graph_viz_path = None
    mock_args.use_surrogate = False
    mock_args.log_level = "INFO"
    mock_parse_args.return_value = mock_args

    with open(mock_args.primitives_file, 'w') as f:
        json.dump(sample_primitives_json_content, f)

    mock_run_pipeline.return_value = {
        'final_state': {'status': 'mocked_complete_direct_call'},
        'telemetry': [{'primitive_id': 'mock_p_direct', 'simulated_duration': 1.0, 'simulated_fidelity': 0.95}],
        'meta_state': {'optimizer_weights': default_test_weights, 'total_executions': 1, 'sum_simulated_fidelity': 0.95, 'num_fidelity_samples': 1}
    }

    with patch('os.path.isfile', return_value=False):
      cli_main()

    mock_parse_args.assert_called_once()
    assert f"Loading primitives from: {mock_args.primitives_file}" in caplog.text

    mock_run_pipeline.assert_called_once()
    call_kwargs = mock_run_pipeline.call_args.kwargs
    assert call_kwargs['code'] == 'LOAD("x.csv")'
    assert isinstance(call_kwargs['catalog'], PrimitiveCatalog)
    assert isinstance(call_kwargs['meta_loop'], MetaCognitiveLoop)
    assert call_kwargs['meta_loop'].learning_rate == 0.05
    assert isinstance(call_kwargs['dsl_compiler'], DSLCompiler)

    assert "Pipeline run completed." in caplog.text
    assert "Final State: {\n  \"status\": \"mocked_complete_direct_call\"\n}" in caplog.text
