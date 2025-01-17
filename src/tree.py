from tree_node import TreeNode

class Tree:
    def __init__(self, root, depth):
        self.root = root
        self.depth = depth
    
    def __copy__(self):
        return Tree(self.root, self.depth)
    
    def get_nodes(self):
        return self.root.get_nodes_from_node()

    def get_non_leaves_nodes(self):
        return self.root.get_non_leaves_nodes_from_node()
    
    def get_leaves_nodes(self):
        return self.root.get_leaves_nodes_from_node()
    
    def validate_tree(self, variables, binary_operators, unary_operators):
        return self.root.validate_tree_from_node(variables, binary_operators, unary_operators)
    
    def evaluate_tree(self, variables, binary_operators, unary_operators):
        return self.root.evaluate_tree_from_node(variables, binary_operators, unary_operators)
    
    def print_tree(self, variables_map):
        self.root.print_tree_from_node(variables_map)

    def print_tree_values(self, variables_map, binary_operators_map, unary_operators_map):
        self.root.print_tree_values_from_node(variables_map, binary_operators_map, unary_operators_map)

    def draw_tree(self):
        self.root.draw_tree_from_node()

    def update_depths_to_root(self, start_node):
        """Update depths from start_node up to root"""
        current = start_node
        while current:
            current.update_depth()
            current = self.get_parent_from_root(current)
    
    def get_max_depth(self):
        """Get the maximum depth of the tree"""
        return self.root.depth if self.root else 0
    

    def get_parent_from_root(self,target):
        """Get the parent of a node from the root"""
        return self.root.get_parent(self.root, target)