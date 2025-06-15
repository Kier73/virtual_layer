import networkx as nx

class ComputationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_counter = 0 # For generating unique node IDs

    def _generate_node_id(self, op_spec: dict) -> str:
        # op_spec is like {'op': 'ADD', 'args': ['x', 'y']}
        # Create a unique ID based on operation index and type for now
        node_id = f"op_{self._node_counter}_{op_spec.get('op', 'Unknown')}"
        self._node_counter += 1
        return node_id

    def add_node(self, op_id: str, **attrs):
        """Adds a node to the graph."""
        self.graph.add_node(op_id, **attrs)
        return op_id

    def add_edge(self, u_of_edge: str, v_of_edge: str):
        """Adds a directed edge from node u to node v."""
        self.graph.add_edge(u_of_edge, v_of_edge)

    def build_from_intent(self, intent: dict):
        """
        Builds the computation graph from an intent dictionary.
        Intent format: {'ops': [{'op': 'OP_NAME', 'args': [...]}, ...], 'vars': {...}}
        """
        self.graph.clear() # Clear any existing graph
        self._node_counter = 0 # Reset node counter for fresh build

        ops = intent.get('ops', [])
        # vars_dict = intent.get('vars', {}) # Not used for dependency yet

        if not ops:
            return

        previous_op_node_id = None

        for op_spec in ops:
            # op_spec example: {'op': 'ADD', 'args': ['x', 10]}
            op_type = op_spec.get('op')
            op_args = op_spec.get('args', [])

            # Generate a unique ID for the operation node
            # For now, using a simple counter combined with op type
            current_op_node_id = self._generate_node_id(op_spec)

            # Add the operation as a node
            # Store operation type and arguments as node attributes
            self.add_node(current_op_node_id, type=op_type, args=op_args, label=f"{op_type}({', '.join(map(str, op_args))})")

            # For now, assume linear dependencies: add an edge from the previous op to the current one.
            if previous_op_node_id:
                self.add_edge(previous_op_node_id, current_op_node_id)

            previous_op_node_id = current_op_node_id

        # Future: Consider variables from intent['vars'] for more complex dependency analysis.
        # For example, if an operation uses a variable defined by 'VAR x = ...',
        # it might depend on the operation that last defined 'x' or an initial state.

    def topological_sort(self) -> list:
        """
        Performs a topological sort of the graph nodes.
        Returns a list of node IDs in topological order.
        """
        if not self.graph:
            return []
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible: # Cycle detected
            # Handle or raise custom error for cyclic graphs if necessary
            raise ValueError("Graph has a cycle, topological sort not possible.")

    def visualize(self, filepath: str = "computation_graph.dot"):
        """
        Visualizes the graph and saves it as a DOT file.
        Requires pydot and graphviz to be installed for nx_pydot.
        Alternatively, can use matplotlib if available.
        """
        if not self.graph:
            print("Graph is empty, nothing to visualize.")
            # Create an empty file or handle as preferred
            with open(filepath, 'w') as f:
                f.write("digraph EmptyGraph {}\n") # Valid empty DOT file
            return

        try:
            # Using nx_pydot to write a DOT file.
            # Ensure node labels are written correctly for visualization.
            # nx.drawing.nx_pydot.write_dot(self.graph, filepath)

            # For better visualization with pydot, especially with node attributes like 'label':
            # Create a Pydot graph from the NetworkX graph
            pydot_graph = nx.drawing.nx_pydot.to_pydot(self.graph)

            # Nodes in NetworkX graph might just be IDs. Attributes are separate.
            # pydot_graph = pydot.Dot(graph_type='digraph')
            # for node, attrs in self.graph.nodes(data=True):
            #     label = attrs.get('label', str(node)) # Use custom label or node ID
            #     pydot_graph.add_node(pydot.Node(str(node), label=label))
            # for u, v in self.graph.edges():
            #     pydot_graph.add_edge(pydot.Edge(str(u), str(v)))

            pydot_graph.write_dot(filepath)
            # print(f"Graph visualized and saved to {filepath}")
        except ImportError:
            print("pydot (and graphviz) not found. Cannot generate DOT file.")
            # Fallback or raise error
            # Could try matplotlib if available:
            # import matplotlib.pyplot as plt
            # try:
            #     pos = nx.spring_layout(self.graph) # Or other layout
            #     nx.draw(self.graph, pos, with_labels=True, labels=nx.get_node_attributes(self.graph, 'label'))
            #     plt.savefig(filepath.replace(".dot", ".png"))
            #     print(f"Graph visualized using matplotlib and saved to {filepath.replace('.dot', '.png')}")
            # except ImportError:
            #     print("matplotlib not found. Visualization skipped.")
            # except Exception as e:
            #     print(f"Error during matplotlib visualization: {e}")
        except Exception as e:
            print(f"An error occurred during DOT file generation: {e}")

if __name__ == '__main__':
    # Example Usage:
    sample_intent_valid = {
        'ops': [
            {'op': 'LOAD', 'args': ['data.csv', 'df1']},
            {'op': 'FILTER', 'args': ['df1', 'colA > 10', 'df2']},
            {'op': 'AGG', 'args': ['df2', 'sum(colB)', 'result_val']},
            {'op': 'SAVE', 'args': ['result_val', 'output.txt']}
        ],
        'vars': {'data_source': 'data.csv'}
    }

    cg = ComputationGraph()
    cg.build_from_intent(sample_intent_valid)

    print("Nodes:", cg.graph.nodes(data=True))
    print("Edges:", cg.graph.edges())

    sorted_nodes = cg.topological_sort()
    print("Topological Sort:", sorted_nodes)

    cg.visualize("example_graph.dot")
    print("Attempted to visualize graph to example_graph.dot")

    # Example with empty ops
    cg_empty = ComputationGraph()
    cg_empty.build_from_intent({'ops': [], 'vars': {}})
    print("\nEmpty graph nodes:", cg_empty.graph.nodes())
    cg_empty.visualize("empty_example_graph.dot")
    print("Attempted to visualize empty graph to empty_example_graph.dot")

    # Example: Graph with a cycle (for testing topological_sort exception)
    # cg_cycle = ComputationGraph()
    # cg_cycle.add_node("A", label="A")
    # cg_cycle.add_node("B", label="B")
    # cg_cycle.add_edge("A", "B")
    # cg_cycle.add_edge("B", "A") # Cycle
    # try:
    #     cg_cycle.topological_sort()
    # except ValueError as e:
    #     print(f"\nCaught expected error for cyclic graph: {e}")
    # cg_cycle.visualize("cycle_graph.dot") # Visualization will work, dot can represent cycles
    # print("Attempted to visualize cyclic graph to cycle_graph.dot")
