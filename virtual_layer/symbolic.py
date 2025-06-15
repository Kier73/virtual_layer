import networkx as nx
import pygraphviz as pgv # For visualization

class ComputationGraph:
    def __init__(self):
        self._g = nx.DiGraph()
        self._node_ids = [] # Helper to store actual node IDs in order of creation

    def build_from_intent(self, intent: dict):
        """
        Builds the computation graph from an intent dictionary.
        Intent format: {'ops': [{'name': 'OP_NAME', 'args': [...]}, ...], 'vars': {...}}
        Note: The key for operation name in intent['ops'] items is expected to be 'name'.
        """
        self._g.clear()
        self._node_ids = [] # Reset node IDs

        ops = intent.get("ops", [])
        if not ops:
            # Changed from raise ValueError to allow empty graphs to be "built" (results in an empty graph)
            # This aligns with potential use cases like an empty DSL script.
            # If an error is desired for no ops, the caller of build_from_intent can check.
            return

        for i, op_dict in enumerate(ops):
            op_name = op_dict.get('name')
            if op_name is None:
                # Or handle this as an error, depending on how strict input validation should be.
                # For now, using a placeholder if 'name' is missing.
                op_name = f"UNKNOWN_OP_{i}"

            # Node ID: use op_name + index to ensure uniqueness
            node_id = f"{op_name}_{i}"
            self._node_ids.append(node_id)

            # Store original name and args as attributes
            # The 'label' attribute will be used by pygraphviz if not specified otherwise.
            # For clarity in visualization, set label to original op_name or op_name(args).
            op_args = op_dict.get('args', [])
            label = f"{op_name}({', '.join(map(str, op_args))})" if op_args else op_name
            self._g.add_node(node_id, name=op_name, args=op_args, label=label)

        # Add edges for a linear chain
        for i in range(len(self._node_ids) - 1):
            self._g.add_edge(self._node_ids[i], self._node_ids[i+1])

    def topological_sort(self) -> list[str]:
        """
        Performs a topological sort of the graph nodes.
        Returns a list of node IDs in topological order.
        Node IDs are in the format "opname_index".
        """
        if not self._g:
            return []
        try:
            return list(nx.topological_sort(self._g))
        except nx.NetworkXUnfeasible: # Cycle detected
            raise ValueError("Graph has a cycle, topological sort not possible.")


    def visualize(self, path: str):
        """
        Visualizes the graph and saves it as a PNG file using pygraphviz.
        """
        if not self._g:
            # Create an empty graph string for pygraphviz or handle appropriately
            agraph = pgv.AGraph(directed=True, strict=True, name="EmptyGraph")
        else:
            # Convert NetworkX graph to AGraph for pygraphviz
            # Node labels should be automatically used if set during add_node.
            # nx_agraph.to_agraph will use the 'label' attribute if present.
            agraph = nx.nx_agraph.to_agraph(self._g)

        agraph.layout(prog='dot') # Layout algorithm (e.g., dot, neato, fdp)
        try:
            agraph.draw(path, format='png')
            # print(f"Graph visualized and saved to {path}")
        except Exception as e:
            # pygraphviz might raise various errors if graphviz tools are not installed
            # or if there are issues with writing the file.
            print(f"Error during graph visualization with pygraphviz: {e}")
            # Consider a fallback or re-raising a custom error.
            # For now, just printing the error.
```
