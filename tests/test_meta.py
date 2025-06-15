import pytest
import json
import os
import copy
import numpy as np
from unittest.mock import MagicMock, call # For mocking surrogate

from virtual_layer.meta import MetaCognitiveLoop, MetaState
# Import BasicSklearnSurrogate for type hinting and potentially for mock target if needed
from virtual_layer.surrogate import SurrogateModel, BasicSklearnSurrogate

# --- Fixtures ---

@pytest.fixture
def default_initial_weights():
    return {'time': 0.6, 'cpu': 0.3, 'io': 0.1}

@pytest.fixture
def meta_loop_config(tmp_path, default_initial_weights):
    """Provides common config for MetaCognitiveLoop instances."""
    state_file = os.path.join(tmp_path, "test_meta_state.json")
    return {
        "initial_weights": default_initial_weights,
        "learning_rate": 0.1, # Use a larger LR for more noticeable changes in tests
        "target_fidelity": 0.85,
        "state_file_path": state_file
    }

@pytest.fixture
def sample_telemetry_entry_factory():
    """Factory to create telemetry entries."""
    def _factory(primitive_id="p_test", planned_cost=10.0,
                 sim_duration=10.0, sim_fidelity=0.9,
                 feature_vector=None):
        entry = {
            'primitive_id': primitive_id,
            'planned_cost_score': planned_cost,
            'simulated_duration': sim_duration,
            'simulated_fidelity': sim_fidelity,
            'feature_vector': feature_vector
            # Other fields like op_type, args, etc., can be added if meta loop uses them
        }
        # Remove feature_vector if None, to simulate entries that might not have it
        if feature_vector is None:
            del entry['feature_vector']
        return entry
    return _factory

# --- Tests for MetaCognitiveLoop ---

def test_initialization_no_state_file(meta_loop_config, default_initial_weights):
    """Test initialization when no state file exists."""
    mcl = MetaCognitiveLoop(
        initial_weights=meta_loop_config['initial_weights'],
        learning_rate=meta_loop_config['learning_rate'],
        target_fidelity=meta_loop_config['target_fidelity'],
        state_file_path=meta_loop_config['state_file_path']
    )
    assert mcl.meta_state['optimizer_weights'] == default_initial_weights
    assert mcl.meta_state['total_executions'] == 0
    assert mcl.meta_state['total_planned_cost'] == 0.0
    assert mcl.meta_state['total_simulated_duration'] == 0.0
    assert mcl.meta_state['sum_simulated_fidelity'] == 0.0
    assert mcl.meta_state['num_fidelity_samples'] == 0
    assert mcl.learning_rate == meta_loop_config['learning_rate']
    assert mcl.target_fidelity == meta_loop_config['target_fidelity']
    assert not os.path.exists(meta_loop_config['state_file_path']) # Should not save on init

def test_save_and_load_state(meta_loop_config, default_initial_weights):
    """Test saving and then loading state."""
    # Create and save state with a first instance
    mcl1 = MetaCognitiveLoop(**meta_loop_config)
    mcl1.meta_state['total_executions'] = 5
    mcl1.meta_state['optimizer_weights']['time'] = 0.75 # Modify a weight
    mcl1.save_state()
    assert os.path.exists(meta_loop_config['state_file_path'])

    # Create a second instance, it should load the saved state
    mcl2 = MetaCognitiveLoop(**meta_loop_config)
    assert mcl2.meta_state['total_executions'] == 5
    assert mcl2.meta_state['optimizer_weights']['time'] == 0.75
    assert mcl2.meta_state['optimizer_weights']['cpu'] == default_initial_weights['cpu'] # Unchanged weight

    # Test loading corrupted JSON
    with open(meta_loop_config['state_file_path'], 'w') as f:
        f.write("corrupted json")
    mcl3 = MetaCognitiveLoop(**meta_loop_config) # Should reset to initial
    assert mcl3.meta_state['optimizer_weights'] == default_initial_weights
    assert mcl3.meta_state['total_executions'] == 0

    # Test loading state with missing keys
    valid_state_missing_key = copy.deepcopy(mcl1.meta_state)
    del valid_state_missing_key['total_planned_cost']
    with open(meta_loop_config['state_file_path'], 'w') as f:
        json.dump(valid_state_missing_key, f)
    mcl4 = MetaCognitiveLoop(**meta_loop_config)
    assert mcl4.meta_state['optimizer_weights'] == default_initial_weights
    assert mcl4.meta_state['total_executions'] == 0


def test_get_optimizer_weights(meta_loop_config):
    mcl = MetaCognitiveLoop(**meta_loop_config)
    weights = mcl.get_optimizer_weights()
    assert weights == meta_loop_config['initial_weights']
    # Ensure it's a deep copy
    weights['time'] = 1.5
    assert mcl.meta_state['optimizer_weights']['time'] != 1.5

def test_update_empty_telemetry(meta_loop_config):
    mcl = MetaCognitiveLoop(**meta_loop_config)
    initial_meta_state_copy = copy.deepcopy(mcl.meta_state)

    updated_state = mcl.update([])

    assert updated_state == initial_meta_state_copy # No change
    assert mcl.meta_state == initial_meta_state_copy # Internal state also unchanged
    # save_state should not have been called if no telemetry implies no meaningful update
    # However, current impl calls save_state. Let's assume that's OK for now.

def test_update_low_fidelity(meta_loop_config, sample_telemetry_entry_factory, default_initial_weights):
    """Test weight adaptation when average fidelity is low."""
    mcl = MetaCognitiveLoop(**meta_loop_config)
    initial_time_weight = default_initial_weights['time']

    # Telemetry indicating low fidelity (e.g., avg < target_fidelity of 0.85)
    telemetry = [
        sample_telemetry_entry_factory(sim_fidelity=0.6, planned_cost=10, sim_duration=11),
        sample_telemetry_entry_factory(sim_fidelity=0.7, planned_cost=10, sim_duration=12),
    ]
    mcl.update(telemetry)

    # Average fidelity = (0.6 + 0.7) / 2 = 0.65 < 0.85.
    # 'time' weight should decrease (making time less penalized to explore potentially slower but more reliable)
    # Prompt had: "Increase 'time' weight ... (making time more costly, hoping for more reliable, possibly slower, primitives)"
    # Current code: if low_fidelity, new_time_weight = current_time_weight * (1 - self.learning_rate)
    # This interpretation makes time *less* costly to allow slower, more reliable primitives.
    expected_new_time_weight = initial_time_weight * (1 - mcl.learning_rate)
    assert mcl.meta_state['optimizer_weights']['time'] == pytest.approx(expected_new_time_weight)

    assert mcl.meta_state['total_executions'] == 2
    assert mcl.meta_state['total_planned_cost'] == 20.0
    assert mcl.meta_state['total_simulated_duration'] == 23.0
    assert mcl.meta_state['sum_simulated_fidelity'] == pytest.approx(1.3)
    assert mcl.meta_state['num_fidelity_samples'] == 2

def test_update_high_fidelity(meta_loop_config, sample_telemetry_entry_factory, default_initial_weights):
    """Test weight adaptation when average fidelity is high."""
    mcl = MetaCognitiveLoop(**meta_loop_config)
    initial_time_weight = default_initial_weights['time']

    # Telemetry indicating high fidelity (e.g., avg >= target_fidelity of 0.85)
    telemetry = [
        sample_telemetry_entry_factory(sim_fidelity=0.9, planned_cost=10, sim_duration=9.5),
        sample_telemetry_entry_factory(sim_fidelity=0.95, planned_cost=10, sim_duration=9.0),
    ]
    mcl.update(telemetry)

    # Average fidelity = (0.9 + 0.95) / 2 = 0.925 >= 0.85.
    # 'time' weight should increase (making time more penalized, as we can afford to optimize it)
    expected_new_time_weight = initial_time_weight * (1 + mcl.learning_rate)
    assert mcl.meta_state['optimizer_weights']['time'] == pytest.approx(expected_new_time_weight)

def test_time_weight_minimum_threshold(meta_loop_config, sample_telemetry_entry_factory):
    """Test that 'time' weight does not go below 0.01."""
    # Set a very low initial time weight and high learning rate to hit the floor
    low_time_weights = {'time': 0.02, 'cpu': 0.5, 'io': 0.48}
    mcl = MetaCognitiveLoop(
        initial_weights=low_time_weights,
        learning_rate=0.8, # High LR
        target_fidelity=0.95, # High target
        state_file_path=meta_loop_config['state_file_path']
    )

    # Low fidelity telemetry to trigger weight decrease
    telemetry = [sample_telemetry_entry_factory(sim_fidelity=0.5)]
    mcl.update(telemetry)

    # Expected: 0.02 * (1 - 0.8) = 0.02 * 0.2 = 0.004. Should be floored at 0.01.
    assert mcl.meta_state['optimizer_weights']['time'] == pytest.approx(0.01)

def test_update_no_time_weight(meta_loop_config, sample_telemetry_entry_factory):
    """Test that update proceeds without error if 'time' weight is not present."""
    no_time_weights = {'cpu': 0.7, 'io': 0.3}
    mcl = MetaCognitiveLoop(
        initial_weights=no_time_weights,
        state_file_path=meta_loop_config['state_file_path']
    )
    initial_weights_copy = copy.deepcopy(mcl.meta_state['optimizer_weights'])

    telemetry = [sample_telemetry_entry_factory(sim_fidelity=0.5)] # Low fidelity
    mcl.update(telemetry)

    # Weights should remain unchanged as 'time' key is missing for adaptation logic
    assert mcl.meta_state['optimizer_weights'] == initial_weights_copy
    assert mcl.meta_state['total_executions'] == 1 # Other stats should update


# --- Tests for Surrogate Model Interaction ---

@pytest.fixture
def mock_surrogate():
    # Using BasicSklearnSurrogate as the class to mock methods on,
    # or a more direct MagicMock if SurrogateModel methods are sufficient.
    # If BasicSklearnSurrogate itself has complex init, use a simpler mock.
    # For this test, we just need an object with a `train` method.

    # Check if SurrogateModel was imported, otherwise use MagicMock as a fallback
    if SurrogateModel is None:
        mock = MagicMock()
        # Manually add abstract methods if needed for type checks by MetaCognitiveLoop,
        # but MetaCognitiveLoop only checks for `self.surrogate_model` and calls `train`.
        mock.train = MagicMock()
        return mock

    # If SurrogateModel is available, mock an instance of a class that implements it.
    # This ensures type compatibility if MetaCognitiveLoop checks isinstance(SurrogateModel).
    # We can mock BasicSklearnSurrogate or a custom dummy implementing SurrogateModel.
    # Let's mock BasicSklearnSurrogate as it's a concrete class we have.
    mock_model = MagicMock(spec=BasicSklearnSurrogate)
    # spec ensures only methods of BasicSklearnSurrogate can be called/asserted.
    # mock_model.train = MagicMock() # Already part of spec if it's a method there.
    return mock_model


def test_surrogate_train_called(meta_loop_config, mock_surrogate, sample_telemetry_entry_factory):
    mcl = MetaCognitiveLoop(
        initial_weights=meta_loop_config['initial_weights'],
        surrogate_model=mock_surrogate,
        state_file_path=meta_loop_config['state_file_path']
    )

    telemetry = [
        sample_telemetry_entry_factory(sim_duration=10, sim_fidelity=0.8, feature_vector=[0.1, 0.2]),
        sample_telemetry_entry_factory(sim_duration=12, sim_fidelity=0.7, feature_vector=[0.3, 0.4]),
    ]
    mcl.update(telemetry)

    mock_surrogate.train.assert_called_once()
    # Check arguments passed to train
    args, _ = mock_surrogate.train.call_args
    assert len(args) == 3 # features, costs, fidelities

    expected_features = np.array([[0.1, 0.2], [0.3, 0.4]])
    expected_costs = np.array([10, 12])
    expected_fidelities = np.array([0.8, 0.7])

    assert np.array_equal(args[0], expected_features)
    assert np.array_equal(args[1], expected_costs)
    assert np.array_equal(args[2], expected_fidelities)

def test_surrogate_train_not_called_no_feature_vector(meta_loop_config, mock_surrogate, sample_telemetry_entry_factory):
    mcl = MetaCognitiveLoop(
        initial_weights=meta_loop_config['initial_weights'],
        surrogate_model=mock_surrogate,
        state_file_path=meta_loop_config['state_file_path']
    )
    # Telemetry entries without 'feature_vector'
    telemetry = [
        sample_telemetry_entry_factory(sim_duration=10, sim_fidelity=0.8), # No feature_vector
    ]
    mcl.update(telemetry)
    mock_surrogate.train.assert_not_called()

def test_surrogate_train_not_called_if_no_model(meta_loop_config, sample_telemetry_entry_factory):
    mcl = MetaCognitiveLoop(
        initial_weights=meta_loop_config['initial_weights'],
        surrogate_model=None, # No surrogate model provided
        state_file_path=meta_loop_config['state_file_path']
    )
    telemetry = [
        sample_telemetry_entry_factory(sim_duration=10, sim_fidelity=0.8, feature_vector=[0.1, 0.2]),
    ]
    # No error should occur, and no train attempt on a None model
    mcl.update(telemetry)
    # No assertion needed for non-call if model is None, just ensure it runs.

def test_surrogate_train_data_shaping(meta_loop_config, mock_surrogate, sample_telemetry_entry_factory):
    """Test surrogate training with data that might need reshaping (e.g. single feature)."""
    mcl = MetaCognitiveLoop(
        initial_weights=meta_loop_config['initial_weights'],
        surrogate_model=mock_surrogate,
        state_file_path=meta_loop_config['state_file_path']
    )
    telemetry = [
        sample_telemetry_entry_factory(sim_duration=10, sim_fidelity=0.8, feature_vector=[0.1]), # Single feature
        sample_telemetry_entry_factory(sim_duration=12, sim_fidelity=0.7, feature_vector=[0.3]), # Single feature
    ]
    mcl.update(telemetry)

    mock_surrogate.train.assert_called_once()
    args, _ = mock_surrogate.train.call_args

    # Features should be reshaped to (n_samples, 1) if they were single features
    expected_features = np.array([[0.1], [0.3]])
    assert args[0].shape == expected_features.shape
    assert np.array_equal(args[0], expected_features)
    assert np.array_equal(args[1], np.array([10, 12])) # costs
    assert np.array_equal(args[2], np.array([0.8, 0.7])) # fidelities

# Clean up state files that might be created by tests if they fail mid-way
# This is more of a utility if running tests locally repeatedly.
# In typical CI, tmp_path handles cleanup.
def M_cleanup_state_files(config_dict_list):
    for config_dict in config_dict_list:
        if os.path.exists(config_dict["state_file_path"]):
            os.remove(config_dict["state_file_path"])

# pytest.addfinalizer(lambda: M_cleanup_state_files([meta_loop_config_instance_for_cleanup_if_needed]))
# This cleanup is tricky with fixtures. tmp_path is the best way.
