import logging
import time
import random
import copy
from typing import List, Tuple, Dict, Any

# Configure basic logging if not already configured by the application
# This is a simple way to ensure logs are output during direct script execution or testing.
# In a larger application, logging is typically configured centrally.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def actuate_sequence(plan_sequence: List[Tuple[Dict[str, Any], float]],
                     initial_state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Simulates the actuation of a sequence of primitives.

    Args:
        plan_sequence (List[Tuple[Dict, float]]): A list of tuples, where each tuple
            contains a primitive dictionary and its pre-calculated cost_score.
            Primitive dict expected keys: 'id', 'op_type', 'args' (and others like 'steps', 'cost_vector').
        initial_state (Dict): The initial state of the system.

    Returns:
        Tuple[Dict, List[Dict]]: A tuple containing the final state of the system
                                 and a list of telemetry entries collected during actuation.
    """
    current_state = copy.deepcopy(initial_state)
    collected_telemetry = []

    if not plan_sequence:
        return current_state, collected_telemetry

    for primitive_dict, cost_score in plan_sequence:
        primitive_id = primitive_dict.get('id', 'unknown_primitive')
        op_type = primitive_dict.get('op_type', 'UNKNOWN_OP_TYPE')
        args = primitive_dict.get('args', [])

        logger.info(f"Executing primitive: {primitive_id} (OpType: {op_type}, Args: {args})")

        # --- Simulate State Update (Stub) ---
        # This is a basic stub. Real implementation would be complex.
        if op_type == 'LOAD':
            if args:
                # Assume first arg is the item to be "loaded" or a variable name representing it
                current_state[str(args[0])] = 'loaded_value_placeholder'
                current_state[f"{args[0]}_status"] = "loaded"
            else:
                current_state['general_load_status'] = "loaded_unspecified_item"
        elif op_type == 'ADD':
            # Example: if 'accumulator' exists, add to it. If not, set a general result.
            # For simplicity, just setting a general result based on args.
            if 'accumulator' in current_state and isinstance(current_state['accumulator'], (int, float)):
                # Try to find a numeric arg to add, or a default value
                val_to_add = 1 # Default if no numeric arg
                for arg in args:
                    if isinstance(arg, (int, float)):
                        val_to_add = arg
                        break
                current_state['accumulator'] += val_to_add
            else:
                 # If args are like ['var1', 'var2'], what does ADD mean?
                 # For this stub, let's assume it means 'var1 + var2' and result is stored.
                 # This is highly dependent on DSL semantics.
                current_state['result_of_add'] = 'computed_sum_placeholder'
        elif op_type == 'SAVE':
            if args:
                current_state[f"{args[0]}_status"] = "saved"
            else:
                current_state['general_save_status'] = "saved_unspecified_item"
        # Add more stub logic for other op_types as needed for tests/demos

        # --- Generate Dummy Telemetry ---
        # Simulate some variation around the planned cost_score for duration
        variation_factor = (0.8 + 0.4 * random.random()) # Range roughly [0.8, 1.2]
        simulated_duration = cost_score * variation_factor

        # Simulate fidelity: perfect if duration matches cost, degrades otherwise, plus some noise
        fidelity_degradation_factor = 0.1 # How much fidelity drops per unit of duration deviation
        duration_deviation = abs(simulated_duration - cost_score)

        simulated_fidelity = 1.0 - (duration_deviation * fidelity_degradation_factor)
        simulated_fidelity -= random.random() * 0.05 # Add some random noise
        simulated_fidelity = max(0.0, min(1.0, simulated_fidelity)) # Clip to [0, 1]

        timestamp = time.time()

        telemetry_entry = {
            'timestamp': timestamp,
            'primitive_id': primitive_id,
            'op_type': op_type, # Adding op_type for clarity in telemetry
            'args': args,       # Adding args for clarity
            'planned_cost_score': cost_score,
            'simulated_duration': simulated_duration,
            'simulated_fidelity': simulated_fidelity,
            'inputs': {}, # Placeholder for actual inputs to the primitive
            'outputs': {}, # Placeholder for actual outputs/results
            'status': 'SUCCESS', # Placeholder, could be 'FAILURE' etc.
            'error_message': None # Placeholder
        }

        logger.info(
            f"Finished primitive: {primitive_id}. "
            f"Duration: {simulated_duration:.2f} (planned: {cost_score:.2f}), "
            f"Fidelity: {simulated_fidelity:.2f}"
        )

        collected_telemetry.append(telemetry_entry)

        # Small delay to simulate work, also makes timestamps different
        # time.sleep(random.uniform(0.001, 0.01))

    return current_state, collected_telemetry


if __name__ == '__main__':
    # Example Usage:
    # Ensure logger is visible for __main__
    logging.getLogger().handlers[0].setLevel(logging.INFO)

    sample_plan = [
        ({'id': 'P_LOAD_data1', 'op_type': 'LOAD', 'args': ['data1'],
          'steps': ['load data1'], 'cost_vector': {'time': 10}},
         10.0), # primitive_dict, cost_score
        ({'id': 'P_ADD_data1_const5', 'op_type': 'ADD', 'args': ['data1', 5],
          'steps': ['add 5 to data1'], 'cost_vector': {'time': 2}},
         2.0),
        ({'id': 'P_SAVE_result', 'op_type': 'SAVE', 'args': ['final_result'],
          'steps': ['save final_result'], 'cost_vector': {'time': 5}},
         5.0)
    ]

    initial_system_state = {'user_id': 123, 'accumulator': 0}

    logger.info("Starting actuation of sample plan...")
    final_state, telemetry_data = actuate_sequence(sample_plan, initial_system_state)

    print("\n--- Final State ---")
    for key, value in final_state.items():
        print(f"{key}: {value}")

    print("\n--- Collected Telemetry ---")
    for i, entry in enumerate(telemetry_data):
        print(f"\nEntry {i+1}:")
        for key, value in entry.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print("\n--- Actuating empty plan ---")
    final_state_empty, telemetry_empty = actuate_sequence([], initial_system_state)
    assert final_state_empty == initial_system_state # Should be a deep copy, but content same
    assert final_state_empty is not initial_system_state # Check it's a copy
    assert telemetry_empty == []
    print("Empty plan actuation completed as expected.")
