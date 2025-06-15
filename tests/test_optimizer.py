import pytest
from virtual_layer.optimizer import PrimitiveCatalog, Optimizer, OptimizationError
from virtual_layer.symbolic import ComputationGraph # For creating graph instances

# --- Fixtures for PrimitiveCatalog ---
@pytest.fixture
def empty_catalog():
    return PrimitiveCatalog()

@pytest.fixture
def populated_catalog():
    catalog = PrimitiveCatalog()
    catalog.register("LOAD", "load_fast", ["lf1"], {'cpu': 10, 'mem': 100})
    catalog.register("LOAD", "load_small", ["ls1"], {'cpu': 20, 'mem': 50})
    catalog.register("ADD", "add_cpu", ["ac1"], {'cpu': 5, 'gpu': 50}) # gpu cost ignored if no weight
    catalog.register("ADD", "add_general", ["ag1"], {'cpu': 10, 'mem': 10})
    catalog.register("SAVE", "save_std", ["ss1"], {'cpu': 8, 'io': 200})
    return catalog

# --- Tests for PrimitiveCatalog ---
def test_catalog_register_single(empty_catalog):
    empty_catalog.register("OP1", "p1_fast", ["s1"], {'cpu': 10})
    candidates = empty_catalog.list_candidates("OP1")
    assert len(candidates) == 1
    assert candidates[0]['id'] == "p1_fast"
    assert candidates[0]['steps'] == ["s1"]
    assert candidates[0]['cost_vector'] == {'cpu': 10}

def test_catalog_register_multiple_same_op(empty_catalog):
    empty_catalog.register("OP1", "p1_fast", ["s1"], {'cpu': 10})
    empty_catalog.register("OP1", "p1_slow", ["s2"], {'cpu': 100})
    candidates = empty_catalog.list_candidates("OP1")
    assert len(candidates) == 2
    ids = {p['id'] for p in candidates}
    assert {"p1_fast", "p1_slow"} == ids

def test_catalog_reregister_updates(empty_catalog):
    empty_catalog.register("OP1", "p1_v1", ["s1"], {'cpu': 10})
    empty_catalog.register("OP1", "p1_v1", ["s1_updated"], {'cpu': 5}) # Re-register with same ID
    candidates = empty_catalog.list_candidates("OP1")
    assert len(candidates) == 1
    assert candidates[0]['steps'] == ["s1_updated"]
    assert candidates[0]['cost_vector']['cpu'] == 5


def test_catalog_list_candidates_exists(populated_catalog):
    candidates = populated_catalog.list_candidates("LOAD")
    assert len(candidates) == 2
    ids = {p['id'] for p in candidates}
    assert {"load_fast", "load_small"} == ids

def test_catalog_list_candidates_not_exists(populated_catalog):
    candidates = populated_catalog.list_candidates("NON_EXISTENT_OP")
    assert len(candidates) == 0

# --- Fixtures for Optimizer and ComputationGraph ---
@pytest.fixture
def optimizer_with_populated_catalog(populated_catalog):
    return Optimizer(populated_catalog)

@pytest.fixture
def empty_comp_graph():
    cg = ComputationGraph()
    cg.build_from_intent({'ops': [], 'vars': {}})
    return cg

@pytest.fixture
def single_node_comp_graph():
    cg = ComputationGraph()
    intent = {'ops': [{'op': 'LOAD', 'args': ['data.csv']}], 'vars': {}}
    cg.build_from_intent(intent)
    return cg

@pytest.fixture
def multi_node_comp_graph():
    cg = ComputationGraph()
    intent = {'ops': [
        {'op': 'LOAD', 'args': ['data.csv']},
        {'op': 'ADD', 'args': ['x', 'y']},
        {'op': 'SAVE', 'args': ['out.csv']}
    ], 'vars': {}}
    cg.build_from_intent(intent)
    return cg

# --- Tests for Optimizer._score_primitive ---
def test_score_primitive(optimizer_with_populated_catalog):
    optimizer = optimizer_with_populated_catalog
    primitive = {'id': 'p1', 'steps': [], 'cost_vector': {'cpu': 10, 'mem': 5}}

    weights1 = {'cpu': 1, 'mem': 1}
    assert optimizer._score_primitive(primitive, weights1) == 15.0

    weights2 = {'cpu': 0.5, 'mem': 2}
    assert optimizer._score_primitive(primitive, weights2) == (0.5*10 + 2*5) # 5 + 10 = 15.0

    weights3 = {'cpu': 1} # mem weight missing, should be treated as 0
    assert optimizer._score_primitive(primitive, weights3) == 10.0

    weights4 = {'cpu': 1, 'mem': 1, 'gpu': 10} # gpu in weights, not in primitive
    assert optimizer._score_primitive(primitive, weights4) == 15.0

    primitive_no_cost = {'id': 'p2', 'steps': [], 'cost_vector': {}}
    assert optimizer._score_primitive(primitive_no_cost, weights1) == 0.0

# --- Tests for Optimizer.optimize ---
def test_optimize_empty_graph(optimizer_with_populated_catalog, empty_comp_graph):
    plan = optimizer_with_populated_catalog.optimize(empty_comp_graph, {'cpu': 1})
    assert plan == []

def test_optimize_single_node_graph(optimizer_with_populated_catalog, single_node_comp_graph):
    # LOAD primitives: load_fast {'cpu': 10, 'mem': 100}, load_small {'cpu': 20, 'mem': 50}

    # Prioritize CPU (load_fast should be chosen)
    weights_cpu = {'cpu': 1, 'mem': 0.1}
    # Score load_fast: 10*1 + 100*0.1 = 10 + 10 = 20
    # Score load_small: 20*1 + 50*0.1 = 20 + 5 = 25
    plan_cpu = optimizer_with_populated_catalog.optimize(single_node_comp_graph, weights_cpu)
    assert len(plan_cpu) == 1
    assert plan_cpu[0][0]['id'] == "load_fast"
    assert plan_cpu[0][1] == pytest.approx(20.0)

    # Prioritize Memory (load_small should be chosen)
    weights_mem = {'cpu': 0.1, 'mem': 1}
    # Score load_fast: 10*0.1 + 100*1 = 1 + 100 = 101
    # Score load_small: 20*0.1 + 50*1 = 2 + 50 = 52
    plan_mem = optimizer_with_populated_catalog.optimize(single_node_comp_graph, weights_mem)
    assert len(plan_mem) == 1
    assert plan_mem[0][0]['id'] == "load_small"
    assert plan_mem[0][1] == pytest.approx(52.0)

def test_optimize_multi_node_graph(optimizer_with_populated_catalog, multi_node_comp_graph):
    # Ops: LOAD, ADD, SAVE
    # LOAD: load_fast (cpu:10,mem:100), load_small (cpu:20,mem:50)
    # ADD: add_cpu (cpu:5,gpu:50), add_general (cpu:10,mem:10)
    # SAVE: save_std (cpu:8,io:200)

    weights = {'cpu': 1, 'mem': 1, 'io': 1, 'gpu':0} # gpu=0 makes add_cpu score 5
    # Expected scores:
    # LOAD: load_fast (10+100=110), load_small (20+50=70) -> choose load_small
    # ADD: add_cpu (5*1 + 50*0 = 5), add_general (10+10=20) -> choose add_cpu
    # SAVE: save_std (8+200=208) -> choose save_std (only one)

    plan = optimizer_with_populated_catalog.optimize(multi_node_comp_graph, weights)
    assert len(plan) == 3
    assert plan[0][0]['id'] == "load_small" # For LOAD
    assert plan[0][1] == pytest.approx(70.0)

    assert plan[1][0]['id'] == "add_cpu"    # For ADD
    assert plan[1][1] == pytest.approx(5.0)

    assert plan[2][0]['id'] == "save_std"   # For SAVE
    assert plan[2][1] == pytest.approx(208.0)

def test_optimize_no_candidates_error(optimizer_with_populated_catalog):
    cg = ComputationGraph()
    # UNKNOWN_OP has no registered primitives in populated_catalog
    intent = {'ops': [{'op': 'UNKNOWN_OP', 'args': []}], 'vars': {}}
    cg.build_from_intent(intent)

    with pytest.raises(OptimizationError, match="No primitive candidates found for operation type 'UNKNOWN_OP'"):
        optimizer_with_populated_catalog.optimize(cg, {'cpu': 1})

def test_optimize_node_missing_type_error(optimizer_with_populated_catalog):
    cg = ComputationGraph()
    # Manually add a node without a 'type' attribute
    cg.add_node("op_0_MALFORMED") # Missing type="...", args="..."

    with pytest.raises(OptimizationError, match="Operation node op_0_MALFORMED is missing 'type' attribute."):
        optimizer_with_populated_catalog.optimize(cg, {'cpu': 1})

def test_optimize_graph_with_cycle_error(optimizer_with_populated_catalog):
    cg = ComputationGraph()
    cg.add_node("op_0_A", type="A", label="A")
    cg.add_node("op_1_B", type="B", label="B")
    cg.add_edge("op_0_A", "op_1_B")
    cg.add_edge("op_1_B", "op_0_A") # Cycle

    # Register dummy primitive for A and B to pass candidate check
    optimizer_with_populated_catalog.catalog.register("A", "pA", [], {'cpu':1})
    optimizer_with_populated_catalog.catalog.register("B", "pB", [], {'cpu':1})

    with pytest.raises(OptimizationError, match="Cannot optimize graph with cycle"):
        optimizer_with_populated_catalog.optimize(cg, {'cpu': 1})
