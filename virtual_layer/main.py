import argparse
import json
import logging
import os
from typing import Optional, List, Dict, Any

# Assuming current directory structure for imports
from .dsl import DSLCompiler, DSLSyntaxError # Using the new regex-based DSLCompiler
# DSLSemanticError is not raised by the new DSLCompiler, but might be by Optimizer
from .symbolic import ComputationGraph
from .optimizer import PrimitiveCatalog, Optimizer, OptimizationError
from .executor import actuate_sequence
from .meta import MetaCognitiveLoop, MetaState # MetaState for type hint
from .surrogate import SurrogateModel, BasicSklearnSurrogate # SurrogateModel for type hint

# Setup basic logging for the entire application
# This ensures that if this module is imported, logging is configured.
# If the module running this (e.g. a script) already configured root logging, this won't override.
if not logging.getLogger(__name__).handlers and not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(code: str,
                 initial_state: Dict[str, Any],
                 catalog: PrimitiveCatalog,
                 meta_loop: MetaCognitiveLoop,
                 dsl_compiler: DSLCompiler,
                 graph_viz_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the full DSL processing pipeline: Parse -> Graph -> Optimize -> Execute -> Update Meta.
    """
    logger.info("Starting pipeline run...")

    # 1. DSL Parsing
    logger.info("Step 1: Parsing DSL code...")
    # The new DSLCompiler returns {'ops': [...], 'vars': {}} where vars is always empty.
    intent_dict = dsl_compiler.parse(code)
    logger.info(f"DSL parsed successfully. Intent: {json.dumps(intent_dict, indent=2)}")

    # 2. Symbolic Graph Generation
    logger.info("Step 2: Building computation graph...")
    graph = ComputationGraph()
    graph.build_from_intent(intent_dict) # build_from_intent uses intent_dict['ops']
    logger.info(f"Computation graph built. Nodes: {graph.graph.number_of_nodes()}, Edges: {graph.graph.number_of_edges()}")
    if graph_viz_path:
        logger.info(f"Visualizing graph to {graph_viz_path}")
        graph.visualize(graph_viz_path) # Requires graphviz installed if not mocked

    # 3. Optimization
    logger.info("Step 3: Optimizing plan...")
    weights = meta_loop.get_optimizer_weights()
    logger.info(f"Using optimizer weights: {weights}")
    optimizer = Optimizer(catalog)
    try:
        # Optimizer expects graph nodes to have 'type' and 'args' attributes.
        # The current ComputationGraph.build_from_intent sets these from the 'ops' list.
        # The 'ops' from regex parser are {'name': op_name, 'args': op_args}.
        # ComputationGraph._generate_node_id uses op_spec.get('op'),
        # and add_node(..., type=op_type, args=op_args).
        # This needs alignment: DSLCompiler produces 'name', Graph expects 'op' for type.
        # For now, let's assume ComputationGraph is adapted or we adapt here.
        # Quick fix: transform intent_dict ops here for now if needed.
        # Or, more robustly, ensure DSLCompiler's output matches ComputationGraph's input expectation.
        # The current ComputationGraph expects op_spec to have 'op' for type.
        # The current regex DSLCompiler output 'name' for op_type.
        # Let's assume ComputationGraph.build_from_intent is flexible or adapted.
        # (Self-correction: Symbolic.py uses op_spec.get('op'), so intent op key should be 'op')
        # The regex compiler produces {'name': ..., 'args': ...}.
        # This means `ComputationGraph.build_from_intent` needs to use `op_spec.get('name')`
        # OR the DSLCompiler output needs to change from 'name' to 'op'.
        # Let's assume DSLCompiler output is the source of truth for 'op name' and map it.
        # No, the optimizer expects graph nodes to have 'type'.
        # The graph builder uses op_spec.get('op').
        # So, the DSL compiler output needs to be {'op': ..., 'args': ...} for each op.
        # This change was missed in the prompt for DSLCompiler. I will make this change in `dsl.py`
        # when I get a chance, or address it here if I can't go back.
        # For now, the tests for DSLCompiler produce {'name':..., 'args':...}.
        # DSLCompiler now outputs {'op': ..., 'args': ...}, which is compatible with ComputationGraph.
        # No workaround needed.
        # graph.build_from_intent(intent_dict) was already called before this try block.

        plan_sequence = optimizer.optimize(graph, weights)
        logger.info(f"Plan optimized. Number of steps in plan: {len(plan_sequence)}")
    except OptimizationError as e:
        logger.error(f"Optimization failed: {e}")
        raise
    except Exception as e: # Catch other unexpected errors during critical steps
        logger.error(f"An unexpected error occurred during graph building or optimization: {e}", exc_info=True)
        raise

    # 4. Execution
    logger.info("Step 4: Executing plan...")
    final_state, telemetry = actuate_sequence(plan_sequence, initial_state)
    logger.info("Plan execution finished.")

    # 5. Meta-Update
    logger.info("Step 5: Updating meta-cognitive loop...")
    updated_meta_state = meta_loop.update(telemetry)
    logger.info(f"Meta-cognitive loop updated. New average fidelity: {meta_loop.get_average_fidelity():.3f}")
    logger.info(f"New optimizer weights: {updated_meta_state['optimizer_weights']}")

    logger.info("Pipeline run completed.")

    return {
        'intent': intent_dict, # Original intent from DSL parse
        'graph_nodes': list(graph.graph.nodes(data=True)), # For easier serialization
        'plan': plan_sequence,
        'final_state': final_state,
        'telemetry': telemetry,
        'meta_state': updated_meta_state
    }

def main():
    parser = argparse.ArgumentParser(description="Virtual Layer Processing Pipeline CLI.")
    parser.add_argument("dsl_code_or_file", type=str, help="DSL code as a string or path to a .dsl file.")
    parser.add_argument("--initial-state-json", type=str, default='{}',
                        help="JSON string for the initial state (default: '{}').")
    parser.add_argument("--initial-weights-json", type=str, default='{"time": 0.6, "cpu": 0.2, "mem": 0.1, "io": 0.1, "quality":0.0}',
                        help="JSON string for initial optimizer weights, used if meta_state file not found.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for meta-cognitive loop.")
    parser.add_argument("--target-fidelity", type=float, default=0.85, help="Target fidelity for meta-cognitive loop.")
    parser.add_argument("--meta-state-file", type=str, default="meta_state.json", help="Path to save/load meta-cognitive state.")
    parser.add_argument("--primitives-file", type=str, default="primitives.json", help="JSON file for primitive catalog.")
    parser.add_argument("--graph-viz-path", type=str, default=None, help="Optional path to save graph visualization (.dot file).")
    parser.add_argument("--use-surrogate", action='store_true', help="Enable BasicSklearnSurrogate model for meta-learning.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level.")

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    logger.info(f"Logging level set to {args.log_level.upper()}")

    dsl_code: str
    if os.path.isfile(args.dsl_code_or_file):
        logger.info(f"Loading DSL code from file: {args.dsl_code_or_file}")
        try:
            with open(args.dsl_code_or_file, 'r') as f:
                dsl_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read DSL file {args.dsl_code_or_file}: {e}")
            return
    else:
        logger.info("Using provided string as DSL code.")
        dsl_code = args.dsl_code_or_file

    try:
        initial_state = json.loads(args.initial_state_json)
        initial_weights = json.loads(args.initial_weights_json)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON input (initial state or weights): {e}")
        return

    # Initialize components
    # The catalog for the regex-based DSLCompiler is a set of known op names.
    # We derive this from the primitives file.
    loaded_primitives_for_catalog_check = {}
    try:
        with open(args.primitives_file, 'r') as f:
            loaded_primitives_for_catalog_check = json.load(f)
    except Exception: # File not found, JSON error, etc.
        pass # Handled by catalog.load_from_json and dummy primitive logic later

    # The DSLCompiler's catalog is just the set of valid operation names.
    dsl_op_catalog = set(loaded_primitives_for_catalog_check.keys())
    if not dsl_op_catalog : # If primitives file was empty or failed to load
         # Fallback for DSL compiler catalog if primitives file is missing/empty
        dsl_op_catalog.update(["LOAD", "ADD", "SAVE", "PROCESS"]) # Add known dummy ops
        logger.info("DSL Compiler using fallback op catalog.")
    else:
        logger.info(f"DSL Compiler catalog initialized with ops: {dsl_op_catalog}")

    dsl_compiler = DSLCompiler(catalog=dsl_op_catalog)

    catalog = PrimitiveCatalog()
    logger.info(f"Loading primitives from: {args.primitives_file}")
    catalog.load_from_json(args.primitives_file)
    if not catalog.primitives:
        logger.warning("Primitive catalog is empty after attempting to load. Using a few dummy primitives for execution.")
        catalog.register("LOAD", "P_LOAD_fallback", ["load_fb"], {"op_type":"LOAD", "time": 10, "cpu": 1, "quality": 0.5})
        catalog.register("ADD", "P_ADD_fallback", ["add_fb"], {"op_type":"ADD", "time": 1, "cpu": 1, "quality": 0.5})
        catalog.register("SAVE", "P_SAVE_fallback", ["save_fb"], {"op_type":"SAVE", "time": 5, "cpu": 1, "quality": 0.5})
        catalog.register("PROCESS", "P_PROCESS_fallback", ["process_fb"], {"op_type":"PROCESS", "time": 7, "cpu": 1, "quality": 0.5})


    surrogate_model: Optional[SurrogateModel] = None
    if args.use_surrogate:
        if BasicSklearnSurrogate and SurrogateModel : # Check if import was successful
            logger.info("Initializing BasicSklearnSurrogate model.")
            surrogate_model = BasicSklearnSurrogate(scale_targets=True)
        else:
            logger.warning("BasicSklearnSurrogate or SurrogateModel base class not available. Proceeding without surrogate model.")

    meta_loop = MetaCognitiveLoop(
        initial_weights=initial_weights,
        learning_rate=args.learning_rate,
        target_fidelity=args.target_fidelity,
        surrogate_model=surrogate_model,
        state_file_path=args.meta_state_file
    )

    try:
        results = run_pipeline(
            code=dsl_code,
            initial_state=initial_state,
            catalog=catalog,
            meta_loop=meta_loop,
            dsl_compiler=dsl_compiler,
            graph_viz_path=args.graph_viz_path
        )

        logger.info("--- Pipeline Results ---")
        logger.info(f"Final State: {json.dumps(results['final_state'], indent=2, default=str)}")
        logger.info(f"Number of Telemetry Entries: {len(results['telemetry'])}")
        if results['telemetry']:
             logger.info(f"Last Telemetry Entry (summary): "
                        f"Primitive={results['telemetry'][-1]['primitive_id']}, "
                        f"Duration={results['telemetry'][-1]['simulated_duration']:.2f}, "
                        f"Fidelity={results['telemetry'][-1]['simulated_fidelity']:.2f}")
        logger.info(f"Final Meta-State: {json.dumps(results['meta_state'], indent=2, default=str)}")
        logger.info(f"Final Average Fidelity: {meta_loop.get_average_fidelity():.3f}")

    except (DSLSyntaxError, OptimizationError) as e:
        logger.error(f"Pipeline execution failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)


if __name__ == '__main__':
    main()
```
