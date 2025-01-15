from graphviz import Digraph
from IPython.display import display

def visualize_tree(root):
    def add_nodes_edges(node, graph):
        """Helper function to add nodes and edges recursively."""
        if not node:
            return
        # Add the current node
        if isinstance(node.value, str):
            if node.coefficient:
                label = f"{round(node.coefficient, 2)}  {node.value}"
                graph.node(str(id(node)), label, shape="box")
            else:
                label = node.value
                graph.node(str(id(node)), label)
        else:
            label = f"{round(node.value, 2)}"
            graph.node(str(id(node)), label, shape="box")

        # Add edges to children
        if node.left:
            graph.edge(str(id(node)), str(id(node.left)), label="L")
            add_nodes_edges(node.left, graph)
        if node.right:
            graph.edge(str(id(node)), str(id(node.right)), label="R")
            add_nodes_edges(node.right, graph)

    # Initialize a directed graph
    dot = Digraph()
    add_nodes_edges(root, dot)
    return dot
