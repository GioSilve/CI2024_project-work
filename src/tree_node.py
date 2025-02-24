import numpy as np
import globals as gb
from print_tree import *
from visualize_tree import visualize_tree, display

class TreeNode:
    def __init__(self, value):
        self.value = value       # This can be an operator or operand
        self.left = None         # Left child
        self.right = None        # Right child
        self.coefficient = None  # multiplicative coefficient for a variable
        self.depth = 1           # Depth of the node
    
    def __copy__(self):
        return TreeNode(self.value, self.left, self.right, self.coefficient)

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return self.value == other.value

    def update_depth(self):
        """Update the depth of this node based on its children"""
        left_depth = self.left.depth if self.left else 0
        right_depth = self.right.depth if self.right else 0
        self.depth = max(left_depth, right_depth) + 1
        
    def update_depths_recursive(self):
        """Update depths for this node and all ancestors"""
        if self.left:
            self.left.update_depths_recursive()
        if self.right:
            self.right.update_depths_recursive()
        self._update_depth()

    def get_nodes_from_node(self):
        """ 
        Returns a list of all nodes in the tree starting from the given node
        """
        nodes = []
        if self:
            nodes.append(self)
        if self.left:
            nodes.extend(self.left.get_nodes_from_node())
        if self.right:
            nodes.extend(self.right.get_nodes_from_node())
        return nodes
    
    def get_non_leaves_nodes_from_node(self):
        """
        Returns a list of all non leaf nodes in the tree starting from the given node
        """
        nodes = []
        if self and (self.left or self.right):
            nodes.append(self)
        if self.left:
            nodes.extend(self.left.get_non_leaves_nodes_from_node())
        if self.right:
            nodes.extend(self.right.get_non_leaves_nodes_from_node())
        return nodes
    
    def get_leaves_nodes_from_node(self):
        """
        Returns a list of all leaf nodes in the tree starting from the given node
        """
        nodes = []
        if self and not (self.left or self.right):
            nodes.append(self)
        if self.left:
            nodes.extend(self.left.get_leaves_nodes_from_node())
        if self.right:
            nodes.extend(self.right.get_leaves_nodes_from_node())
        return nodes
        
    def validate_tree_from_node(self):
        """
        Returns True if the tree is syntactically correct without checking domain constraints of operators, False otherwise
        """
        if not self:
            return True
        
        if self.value in gb.BINARY_OPERATORS:
            if not self.left or not self.right:
                return False  # Operators must have two children
            return self.left.validate_tree_from_node() and self.right.validate_tree_from_node()
        
        elif self.value in gb.UNARY_OPERATORS:  # Allow unary operators
            if self.right and not self.left:
                return self.right.validate_tree_from_node()
            return False  # Unary operators must have one child on the right
        
        # Allow variables
        elif self.value in gb.VARIABLES_MAP:
            return True
        elif isinstance(self.value, float):
            return True
        else:
            return False  # Invalid value
    
    def evaluate_tree_from_node(self):
        """
        Returns the value of the expression represented by the tree starting form a specific node
        """
        if not self:
            raise ValueError("Cannot evaluate an empty tree.")
        
        # Check if it's a binary operator
        if self.value in gb.BINARY_OPERATORS:
            left_val = self.left.evaluate_tree_from_node()
            right_val = self.right.evaluate_tree_from_node()
                # self.draw_tree_from_node()
            restult = gb.BINARY_OPERATORS[self.value](left_val, right_val)
            if np.any(np.isnan(restult)):
                print("invalid found")
            return restult
        # Check if it's a unary operator
        elif self.value in gb.UNARY_OPERATORS:
            right_val = self.right.evaluate_tree_from_node() # Typically applies to right child
            res = gb.UNARY_OPERATORS[self.value](right_val)                
            return  res # Correct unary application

        # Check if it's a variable
        elif self.value in gb.VARIABLES_MAP:
            return np.multiply(self.coefficient, gb.VARIABLES_MAP[self.value])  # Lookup the variable value
        
        # Check if it's a numeric constant or coefficient
        elif isinstance(self.value, (int, float)):
            return self.value  # Return as-is for numeric leaf selfs
        
        # If none of the above, it's an error
        else:
            raise ValueError(f"Invalid self value: {self.value}")

    def print_tree_from_node(self):
        print_expr(self)

    def print_tree_values_from_node(self):
        print_expr_values(self)

    def draw_tree_from_node(self):
        display(visualize_tree(self))

    def get_parent(self, root, target):
        """
        Find the parent of a target node in the tree.
        """
        # The 'is' operator is unaffected by __eq__ and always checks if two variables refer to the same object in memory.
        if not root or root is target:
            return None
        if root.left is target or root.right is target:
            return root
        return self.get_parent(root.right, target) or self.get_parent(root.left, target)