from virtual_layer.dsl import DSLCompiler
from virtual_layer.symbolic import ComputationGraph
from virtual_layer.optimizer import PrimitiveCatalog, Optimizer
from virtual_layer.surrogate import SurrogateModel
from virtual_layer.executor import actuate_sequence
from virtual_layer.meta import MetaCognitiveLoop
from typing import Any, List


def main(dsl_code: str, initial_state: Any, catalogs: PrimitiveCatalog, weights: List[float]):
    # 1. Parse DSL into symbolic intent
    compiler = DSLCompiler(primitives_catalog=catalogs.mapping)
    intent = compiler.parse(dsl_code)

    # 2. Build computation graph
    graph = ComputationGraph()
    graph.build_from_intent(intent)

    # 3. Optimize to physical plan
    optimizer = Optimizer(catalogs, weights)
    plan = optimizer.optimize(graph)

    # 4. Optionally use surrogates
    # surrogate = SurrogateModel('example')

    # 5. Execute plan
    final_state = actuate_sequence(plan, initial_state)

    # 6. Meta-cognitive update
    meta = MetaCognitiveLoop(weights)
    # meta.update({'final_state': final_state})

    return final_state
