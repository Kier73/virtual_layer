import pytest
import logging
import time
import copy
from virtual_layer.executor import actuate_sequence

# --- Fixture for a sample plan_sequence ---
@pytest.fixture
def sample_plan_sequence():
    """Provides a sample plan_sequence for testing."""
    plan = [
        ({'id': 'P_LOAD_dataA', 'op_type': 'LOAD', 'args': ['dataA'],
          'steps': ['load dataA from source'], 'cost_vector': {'time': 10, 'cpu': 5}},
         10.0), # (primitive_dict, cost_score)
        ({'id': 'P_ADD_val5', 'op_type': 'ADD', 'args': ['dataA_processed', 5],
          'steps': ['add constant 5'], 'cost_vector': {'time': 2, 'cpu': 2}},
         2.0),
        ({'id': 'P_SAVE_output', 'op_type': 'SAVE', 'args': ['final_output'],
          'steps': ['save to destination'], 'cost_vector': {'time': 5, 'io': 15}},
         5.0)
    ]
    return plan

@pytest.fixture
def initial_state_fixture():
    """Provides a consistent initial state."""
    return {'user_id': 'test_user', 'files_processed': 0, 'accumulator': 100}

# --- Tests for actuate_sequence ---

def test_actuate_empty_sequence(initial_state_fixture, caplog):
    """Test with an empty plan_sequence."""
    caplog.set_level(logging.INFO)
    initial_state_copy = copy.deepcopy(initial_state_fixture)

    final_state, collected_telemetry = actuate_sequence([], initial_state_fixture)

    assert final_state == initial_state_copy # Should be identical in content
    assert final_state is not initial_state_fixture # Should be a deep copy
    assert collected_telemetry == []

    # Check that no execution logs were made for primitives
    for record in caplog.records:
        assert "Executing primitive" not in record.message
        assert "Finished primitive" not in record.message

def test_actuate_single_primitive(initial_state_fixture, caplog):
    """Test with a single primitive in the plan_sequence."""
    caplog.set_level(logging.INFO)
    primitive_load = {
        'id': 'P_LOAD_itemX', 'op_type': 'LOAD', 'args': ['itemX'],
        'steps': ['load itemX'], 'cost_vector': {'time': 5}
    }
    cost_score_load = 5.0
    plan = [(primitive_load, cost_score_load)]

    initial_state_copy = copy.deepcopy(initial_state_fixture)
    final_state, collected_telemetry = actuate_sequence(plan, initial_state_fixture)

    # Verify state update
    assert final_state['itemX'] == 'loaded_value_placeholder'
    assert final_state['itemX_status'] == 'loaded'
    assert final_state['user_id'] == initial_state_copy['user_id'] # Unrelated state preserved

    # Verify telemetry
    assert len(collected_telemetry) == 1
    telemetry_entry = collected_telemetry[0]
    assert telemetry_entry['primitive_id'] == 'P_LOAD_itemX'
    assert telemetry_entry['op_type'] == 'LOAD'
    assert telemetry_entry['args'] == ['itemX']
    assert telemetry_entry['planned_cost_score'] == cost_score_load
    assert isinstance(telemetry_entry['simulated_duration'], float)
    assert 0.0 <= telemetry_entry['simulated_fidelity'] <= 1.0
    assert isinstance(telemetry_entry['timestamp'], float)
    assert telemetry_entry['status'] == 'SUCCESS'

    # Check logs
    assert len(caplog.records) == 2 # Start and Finish log messages
    assert f"Executing primitive: {primitive_load['id']}" in caplog.records[0].message
    assert f"Finished primitive: {primitive_load['id']}" in caplog.records[1].message
    assert f"Duration: {telemetry_entry['simulated_duration']:.2f}" in caplog.records[1].message
    assert f"Fidelity: {telemetry_entry['simulated_fidelity']:.2f}" in caplog.records[1].message

def test_actuate_multiple_primitives(sample_plan_sequence, initial_state_fixture, caplog):
    """Test with multiple primitives, checking cumulative state and telemetry."""
    caplog.set_level(logging.INFO)
    num_primitives = len(sample_plan_sequence)

    final_state, collected_telemetry = actuate_sequence(sample_plan_sequence, initial_state_fixture)

    # Verify final state (check some expected changes)
    # Based on stub logic in actuate_sequence:
    # LOAD 'dataA'
    assert final_state['dataA'] == 'loaded_value_placeholder'
    assert final_state['dataA_status'] == "loaded"
    # ADD (modifies 'accumulator' due to initial_state_fixture having it)
    # Initial accumulator = 100, P_ADD_val5 adds 5
    assert final_state['accumulator'] == 105
    # SAVE 'final_output'
    assert final_state['final_output_status'] == "saved"
    # Check initial state is still there
    assert final_state['user_id'] == initial_state_fixture['user_id']

    # Verify telemetry
    assert len(collected_telemetry) == num_primitives
    for i, entry in enumerate(collected_telemetry):
        expected_primitive_dict, expected_score = sample_plan_sequence[i]
        assert entry['primitive_id'] == expected_primitive_dict['id']
        assert entry['op_type'] == expected_primitive_dict['op_type']
        assert entry['args'] == expected_primitive_dict['args']
        assert entry['planned_cost_score'] == expected_score
        assert isinstance(entry['simulated_duration'], float)
        assert 0.0 <= entry['simulated_fidelity'] <= 1.0

    # Check logs
    assert len(caplog.records) == num_primitives * 2 # Start and Finish for each
    for i in range(num_primitives):
        primitive_id = sample_plan_sequence[i][0]['id']
        assert f"Executing primitive: {primitive_id}" in caplog.records[i*2].message
        assert f"Finished primitive: {primitive_id}" in caplog.records[i*2+1].message

def test_initial_state_not_modified(sample_plan_sequence, initial_state_fixture):
    """Ensure the original initial_state dictionary is not modified."""
    initial_state_original_copy = copy.deepcopy(initial_state_fixture)

    actuate_sequence(sample_plan_sequence, initial_state_fixture)

    # Compare the passed initial_state_fixture with its original deep copy
    assert initial_state_fixture == initial_state_original_copy

def test_telemetry_values_within_expected_ranges(sample_plan_sequence, initial_state_fixture):
    """Check if simulated duration and fidelity are within sensible bounds."""
    _, collected_telemetry = actuate_sequence(sample_plan_sequence, initial_state_fixture)

    for entry in collected_telemetry:
        cost_score = entry['planned_cost_score']
        duration = entry['simulated_duration']
        fidelity = entry['simulated_fidelity']

        # Duration should be positive and somewhat related to cost_score
        # (0.8 * cost_score) to (1.2 * cost_score) roughly
        assert duration > 0
        assert cost_score * 0.7 < duration < cost_score * 1.3 # Allow slight extra margin for random

        # Fidelity must be between 0 and 1
        assert 0.0 <= fidelity <= 1.0

def test_actuate_primitive_with_no_args(initial_state_fixture, caplog):
    """Test with a primitive that might have no arguments specified."""
    caplog.set_level(logging.INFO)
    primitive_no_args = {
        'id': 'P_PROCESS_all', 'op_type': 'PROCESS', # op_type not in current stub, but fine
        'steps': ['process all items'], 'cost_vector': {'time': 7}
        # 'args' key is missing
    }
    cost_score_no_args = 7.0
    plan = [(primitive_no_args, cost_score_no_args)]

    final_state, collected_telemetry = actuate_sequence(plan, initial_state_fixture)

    # Check telemetry for default args value
    assert len(collected_telemetry) == 1
    telemetry_entry = collected_telemetry[0]
    assert telemetry_entry['primitive_id'] == 'P_PROCESS_all'
    assert telemetry_entry['op_type'] == 'PROCESS'
    assert telemetry_entry['args'] == [] # Default args should be empty list

    # Check logs
    assert f"Args: []" in caplog.records[0].message # Check how no args are logged
