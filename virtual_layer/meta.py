import json
import logging
import os
import copy
from typing import Optional, List, Dict, Any, TypedDict

import numpy as np # Required for surrogate model interaction

# Assuming surrogate module and its classes will be in the same directory or accessible
try:
    from .surrogate import SurrogateModel, BasicSklearnSurrogate
except ImportError: # Handle cases where this might be run standalone or during early test phases
    SurrogateModel = None
    BasicSklearnSurrogate = None

# Configure basic logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetaState(TypedDict):
    optimizer_weights: Dict[str, float]
    total_executions: int
    total_planned_cost: float
    total_simulated_duration: float
    # total_simulated_fidelity: float # Sum of fidelities to calculate average
    # Instead of total_simulated_fidelity, let's store sum and count for average
    sum_simulated_fidelity: float
    num_fidelity_samples: int # Count of telemetry entries that had fidelity
    # average_fidelity: float # Calculated on the fly or when needed


class MetaCognitiveLoop:
    def __init__(self,
                 initial_weights: Dict[str, float],
                 learning_rate: float = 0.01,
                 target_fidelity: float = 0.85,
                 surrogate_model: Optional[SurrogateModel] = None,
                 state_file_path: str = "meta_state.json"):

        self.learning_rate: float = learning_rate
        self.target_fidelity: float = target_fidelity
        self.surrogate_model: Optional[SurrogateModel] = surrogate_model
        self.state_file_path: str = state_file_path

        self.meta_state: MetaState = self._load_state(initial_weights)

    def _load_state(self, initial_weights: Dict[str, float]) -> MetaState:
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    loaded_data = json.load(f)
                    # Basic validation if all keys are present
                    required_keys = ['optimizer_weights', 'total_executions', 'total_planned_cost',
                                     'total_simulated_duration', 'sum_simulated_fidelity', 'num_fidelity_samples']
                    if all(key in loaded_data for key in required_keys):
                        logger.info(f"MetaCognitiveLoop state loaded from {self.state_file_path}")
                        return loaded_data # type: ignore # Trusting file content for now
                    else:
                        logger.warning(f"State file {self.state_file_path} is missing required keys. Initializing with defaults.")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {self.state_file_path}. Initializing with default state.")
            except Exception as e:
                logger.error(f"Unexpected error loading state from {self.state_file_path}: {e}. Initializing with default state.")

        logger.info("MetaCognitiveLoop initializing new state.")
        return MetaState(
            optimizer_weights=copy.deepcopy(initial_weights), # Ensure we don't modify the passed dict
            total_executions=0,
            total_planned_cost=0.0,
            total_simulated_duration=0.0,
            sum_simulated_fidelity=0.0,
            num_fidelity_samples=0
        )

    def save_state(self):
        try:
            with open(self.state_file_path, 'w') as f:
                json.dump(self.meta_state, f, indent=4)
            logger.info(f"MetaCognitiveLoop state saved to {self.state_file_path}")
        except IOError:
            logger.error(f"Error saving state to {self.state_file_path}")
        except Exception as e:
            logger.error(f"Unexpected error saving state: {e}")


    def get_optimizer_weights(self) -> Dict[str, float]:
        return copy.deepcopy(self.meta_state['optimizer_weights']) # Return a copy

    def get_average_fidelity(self) -> float:
        if self.meta_state['num_fidelity_samples'] == 0:
            return 0.0 # Or some default, e.g. target_fidelity to avoid aggressive first adaptation
        return self.meta_state['sum_simulated_fidelity'] / self.meta_state['num_fidelity_samples']

    def update(self, executed_plan_telemetry: List[Dict[str, Any]]) -> MetaState:
        if not executed_plan_telemetry:
            logger.info("MetaCognitiveLoop update called with no telemetry. No changes made.")
            return copy.deepcopy(self.meta_state)

        current_batch_planned_cost = 0.0
        current_batch_simulated_duration = 0.0
        current_batch_sum_fidelity = 0.0
        num_fidelity_entries_in_batch = 0

        features_for_surrogate: List[np.ndarray] = []
        costs_for_surrogate: List[float] = []
        fidelities_for_surrogate: List[float] = []

        for entry in executed_plan_telemetry:
            current_batch_planned_cost += entry.get('planned_cost_score', 0.0)
            current_batch_simulated_duration += entry.get('simulated_duration', 0.0)

            fidelity = entry.get('simulated_fidelity')
            if fidelity is not None: # Ensure fidelity was reported
                current_batch_sum_fidelity += fidelity
                num_fidelity_entries_in_batch +=1

            # Collect data for surrogate model if available
            if 'feature_vector' in entry and fidelity is not None and entry.get('simulated_duration') is not None:
                # Assuming feature_vector is already a np.ndarray or compatible list
                features_for_surrogate.append(np.array(entry['feature_vector']))
                costs_for_surrogate.append(entry['simulated_duration'])
                fidelities_for_surrogate.append(fidelity)

        # Update totals in meta_state
        self.meta_state['total_executions'] += len(executed_plan_telemetry)
        self.meta_state['total_planned_cost'] += current_batch_planned_cost
        self.meta_state['total_simulated_duration'] += current_batch_simulated_duration
        if num_fidelity_entries_in_batch > 0:
            self.meta_state['sum_simulated_fidelity'] += current_batch_sum_fidelity
            self.meta_state['num_fidelity_samples'] += num_fidelity_entries_in_batch

        overall_average_fidelity = self.get_average_fidelity()
        logger.info(f"Meta-update: Overall average fidelity = {overall_average_fidelity:.3f} (target: {self.target_fidelity:.3f})")

        # --- Weight Adaptation Logic (Simple Example) ---
        # Focus on 'time' weight for now, if it exists.
        # This is a very basic adaptation strategy.
        if 'time' in self.meta_state['optimizer_weights']:
            current_time_weight = self.meta_state['optimizer_weights']['time']
            if overall_average_fidelity < self.target_fidelity and self.meta_state['num_fidelity_samples'] > 0:
                # If fidelity is low, make 'time' more important (increase its weight)
                # This assumes that higher 'time' cost primitives might be more reliable/higher fidelity.
                # Or, it means we want to spend more "time resource" to achieve fidelity.
                # This logic might need to be more nuanced based on primitive characteristics.
                # For now, increasing time weight makes time-expensive primitives less attractive.
                # If the goal is to pick *more reliable* (higher fidelity) primitives,
                # and if those tend to be *slower* (higher time cost), then making time
                # *less* of a penalty (decreasing its weight) might choose them.
                # Let's assume for now: low fidelity -> try to be more conservative,
                # which might mean picking things that are "less costly" in some critical way.
                # If 'time' is a proxy for 'effort', increasing its weight makes us avoid 'high effort' things.
                # This is a placeholder for more sophisticated logic.
                # For this example: if fidelity is low, perhaps we are too focused on speed.
                # Let's try to value "non-time" costs more, or make "time" less important to explore other options.
                # Alternative: If low fidelity, make 'time' *more* important (costly),
                # to select primitives that are faster, assuming faster ones are simpler and more reliable.
                # This is very domain specific. Let's stick to the prompt's hint:
                # "Increase 'time' weight ... (making time more costly, hoping for more reliable, possibly slower, primitives)"
                # This implies an inverse relationship: higher time cost = higher reliability.
                # So if fidelity is low, we should choose primitives that are *allowed* to take more time.
                # This means the *penalty* for time should *decrease*.
                new_time_weight = current_time_weight * (1 - self.learning_rate) # Decrease penalty for time
                logger.info(f"Low fidelity ({overall_average_fidelity:.3f}), decreasing 'time' weight from {current_time_weight:.3f} to {new_time_weight:.3f}")

            elif self.meta_state['num_fidelity_samples'] > 0 : # Fidelity is good or target met
                # If fidelity is good, we can afford to optimize more for 'time' (make it relatively more penalized)
                new_time_weight = current_time_weight * (1 + self.learning_rate) # Increase penalty for time
                logger.info(f"Good fidelity ({overall_average_fidelity:.3f}), increasing 'time' weight from {current_time_weight:.3f} to {new_time_weight:.3f}")
            else: # No fidelity samples yet, or target met with no samples (no change)
                new_time_weight = current_time_weight

            self.meta_state['optimizer_weights']['time'] = max(0.01, new_time_weight) # Ensure weight stays positive

            # Normalization (optional, if weights should sum to a constant, e.g., 1)
            # total_weight = sum(self.meta_state['optimizer_weights'].values())
            # if total_weight > 0:
            #    self.meta_state['optimizer_weights'] = {k: v / total_weight for k, v in self.meta_state['optimizer_weights'].items()}
        else:
            logger.warning("'time' weight not found in optimizer_weights. Skipping weight adaptation for 'time'.")


        # --- Surrogate Model Retraining ---
        if self.surrogate_model and features_for_surrogate:
            if SurrogateModel is None: # Check if import failed
                logger.error("SurrogateModel class not available (ImportError). Cannot retrain.")
            else:
                try:
                    # Ensure data is in correct numpy array format
                    np_features = np.array(features_for_surrogate)
                    np_costs = np.array(costs_for_surrogate)
                    np_fidelities = np.array(fidelities_for_surrogate)

                    # Basic check for sufficient data shape (e.g., at least one sample, features match)
                    if np_features.ndim == 1: # Single feature, multiple samples
                        np_features = np_features.reshape(-1,1)
                    elif np_features.ndim == 0 and np_features.size > 0 : # Single sample, single feature
                         np_features = np_features.reshape(1,1)


                    if np_costs.ndim == 0: np_costs = np_costs.reshape(-1)
                    if np_fidelities.ndim == 0: np_fidelities = np_fidelities.reshape(-1)

                    if np_features.shape[0] > 0 and \
                       np_features.shape[0] == np_costs.shape[0] and \
                       np_features.shape[0] == np_fidelities.shape[0]:

                        logger.info(f"Retraining surrogate model with {np_features.shape[0]} new data points.")
                        self.surrogate_model.train(np_features, np_costs, np_fidelities)
                        logger.info("Surrogate model retraining attempt complete.")
                    else:
                        logger.warning(f"Insufficient or mismatched data for surrogate retraining. "
                                     f"Features shape: {np_features.shape}, "
                                     f"Costs shape: {np_costs.shape}, "
                                     f"Fidelities shape: {np_fidelities.shape}")
                except Exception as e:
                    logger.error(f"Error during surrogate model retraining: {e}")

        self.save_state()
        return copy.deepcopy(self.meta_state)

if __name__ == '__main__':
    # Example Usage
    initial_test_weights = {'time': 0.5, 'cpu': 0.3, 'io': 0.2}

    # Test without surrogate first
    meta_loop = MetaCognitiveLoop(initial_weights=initial_test_weights, state_file_path="meta_test_state.json")
    print("Initial MetaState:", meta_loop.meta_state)

    # Simulate some telemetry
    telemetry1 = [
        {'primitive_id': 'p1', 'planned_cost_score': 10, 'simulated_duration': 12, 'simulated_fidelity': 0.7, 'feature_vector': [1,2,3]},
        {'primitive_id': 'p2', 'planned_cost_score': 5, 'simulated_duration': 4.5, 'simulated_fidelity': 0.9, 'feature_vector': [4,5,6]},
    ]
    meta_loop.update(telemetry1)
    print("MetaState after update 1:", meta_loop.meta_state)
    print(f"Avg Fidelity after update 1: {meta_loop.get_average_fidelity()}")


    telemetry2 = [
        {'primitive_id': 'p3', 'planned_cost_score': 8, 'simulated_duration': 7, 'simulated_fidelity': 0.95, 'feature_vector': [0.1,0.2,0.3]},
    ]
    meta_loop.update(telemetry2)
    print("MetaState after update 2:", meta_loop.meta_state)
    print(f"Avg Fidelity after update 2: {meta_loop.get_average_fidelity()}")

    # Test with a mock surrogate (if BasicSklearnSurrogate is available)
    if BasicSklearnSurrogate and SurrogateModel:
        class MockSurrogate(SurrogateModel):
            def __init__(self): self.trained_with_data = None
            def train(self, features, costs, fidelities):
                print(f"MockSurrogate: train() called with {features.shape[0]} samples.")
                self.trained_with_data = (features, costs, fidelities)
            def predict_cost(self, features): return np.zeros(features.shape[0])
            def predict_fidelity(self, features): return np.ones(features.shape[0])

        mock_surrogate_model = MockSurrogate()
        meta_loop_with_surrogate = MetaCognitiveLoop(
            initial_weights=initial_test_weights,
            surrogate_model=mock_surrogate_model,
            state_file_path="meta_surrogate_test_state.json"
        )
        meta_loop_with_surrogate.update(telemetry1) # telemetry1 has feature_vectors
        assert mock_surrogate_model.trained_with_data is not None
        print("MetaState with surrogate after update:", meta_loop_with_surrogate.meta_state)
        # Clean up test state file
        if os.path.exists("meta_test_state.json"): os.remove("meta_test_state.json")
        if os.path.exists("meta_surrogate_test_state.json"): os.remove("meta_surrogate_test_state.json")
