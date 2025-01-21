import numpy as np
import random
import globals as gb
from generators import compute_coefficient, generate_safe_constant
from utils import are_compatible, get_unary_weights
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
    
    def validate_tree(self):
        return self.root.validate_tree_from_node()
    
    def evaluate_tree(self):
        return self.root.evaluate_tree_from_node()
    
    def print_tree(self):
        self.root.print_tree_from_node()

    def print_tree_values(self):
        self.root.print_tree_values_from_node()

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


def random_initial_tree(depth, maxdepth, variables):
    """
    Recursively generates a random tree structure with specified depth constraints.

    The function constructs a tree by randomly adding variables, unary operators, or binary operators
    based on the current depth relative to the maximum depth allowed. At the maximum depth, variables 
    or constants are added as leaf nodes. At one level before the maximum depth, unary operators are 
    added. At other levels, binary operators are added.

    Args:
        depth (int): The current depth of the tree being constructed.
        maxdepth (int): The maximum depth allowed for the tree.
        variables (list): A list of variables available to be used as leaf nodes.

    Returns:
        TreeNode: The root node of the generated tree or subtree.
    """
    if depth == maxdepth:  # Add a variable until they are all chosen, if yes add a number
        if variables:
            var = random.choice(variables)
            leaf = TreeNode(var)
            leaf.coefficient = compute_coefficient(var, gb.VARIABLES_MAP, gb.COEFFICIENT_RANGES)
            variables.remove(var)
        else:
            leaf = TreeNode(generate_safe_constant(gb.Y))
            leaf.coefficient = 1
        leaf.depth = 1
        return leaf
    
    elif depth == maxdepth - 1: # Add a unary operator
        node = TreeNode(None)
        node.right = random_initial_tree(depth + 1, maxdepth, variables)
        node.left = None
    
        available_unary = [op for op in gb.UNARY_OPERATORS if are_compatible(op, np.multiply(gb.VARIABLES_MAP[node.right.value], node.right.coefficient) if node.right.value in gb.VARIABLES_MAP else node.right.value)]
        available_weights = get_unary_weights(available_unary)
        node.value = np.random.choice(available_unary, p=available_weights) # If a choice of a variant unary operator was made, choose a random variant from all the possible ones
        node.update_depth()
        return node
    
    else: # Add a binary operator
        node = TreeNode(None)
        node.left = random_initial_tree(depth + 1, maxdepth, variables)
        node.right = random_initial_tree(depth + 1, maxdepth, variables)
        available_binary = [op for op in gb.BINARY_OPERATORS if are_compatible(op, node.right.evaluate_tree_from_node(), node.left.evaluate_tree_from_node())]
        node.value = random.choice(available_binary) # Choose a random binary operator from all the possible ones
        node.update_depth()
        return node
    

def get_random_leaf():
    """
    Returns a random leaf node that is either a variable or a constant.

    The leaf node is chosen with equal probability to be either a variable or a constant.
    If a variable is chosen, the coefficient is computed using the compute_coefficient function.
    If a constant is chosen, it is generated using the generate_safe_constant function.

    Returns:
        TreeNode: The randomly generated leaf node.
    """
    if random.choice([0, 1]):
        random_leaf = TreeNode(random.choice(list(gb.VARIABLES_MAP.keys())))
        random_leaf.coefficient = compute_coefficient(random_leaf.value, gb.VARIABLES_MAP, gb.COEFFICIENT_RANGES)
    else:
        random_leaf = TreeNode(generate_safe_constant(gb.Y))
        random_leaf.coefficient = 1
    return random_leaf
    

def validate_after_replacement(root:TreeNode, replaced_node: Tree):
    """
    Validate the tree after replacing a subtree.

    Args:
        root (TreeNode): The root of the tree.
        replaced_node (TreeNode): The node that was replaced.
        gb.unary_operators (list): List of unary operators.
        gb.binary_operators (list): List of binary operators.

    Returns:
        bool: True if the tree is valid, False otherwise.
    """
    
    def is_valid_node(parent: TreeNode) -> bool:
        right_val =  parent.right.evaluate_tree_from_node()
        if parent.value in gb.UNARY_OPERATORS.keys():
            return are_compatible(parent.value, right_val)
        
        elif parent.value in gb.BINARY_OPERATORS.keys():
            left_val = parent.left.evaluate_tree_from_node()
            return are_compatible(parent.value, right_val, left_val)
        return True

    current = replaced_node
    while current:
        parent = root.get_parent(root, current)
        if not parent:
            break

        if not is_valid_node(parent):
            return False
        current = parent  # Traverse up to the root

    return True
    

def swap_subtrees(source_tree, target_tree):
    """
    Try to swap subtrees from source_tree to target_tree.

    Args:
        source_tree (Tree): The source tree.
        target_tree (Tree): The target tree.

    Returns:
        bool: True if a swap was successful, False otherwise.
    """
    source_nodes = source_tree.get_nodes()

    while source_nodes:
        source_node = random.choice(source_nodes)
        if try_swap(source_node, target_tree):
            return True
        source_nodes.remove(source_node)
    return False


def try_swap(source_node: TreeNode, target_tree: Tree, filter_leaves_parents=False):
    """
    Attempt to swap a subtree from the source node into the target tree.

    This function tries to replace a node in the target tree with the source node,
    ensuring that the resulting tree remains valid by using compatibility checks and
    validation after replacement.

    Args:
        source_node (TreeNode): The root node of the subtree to be swapped into the target tree.
        target_tree (Tree): The tree where the source_node will be potentially inserted.
        filter_leaves_parents (bool, optional): If set to True, only consider target nodes whose
            children are not leaves to avoid making a leaf a parent. Defaults to False.

    Returns:
        bool: True if the swap was successful and the tree remains valid, False otherwise.
    """
    target_nodes = target_tree.get_non_leaves_nodes()
    if filter_leaves_parents:
        # Consider only nodes whose children are not leaves to avoid adding subtree to leaf in mutation
        target_nodes = [node for node in target_nodes if node.right.right or node.right.left]
    
    while target_nodes:
        # target_node -> parent del nodo in target_tree che verrà sostituito con source_tree
        target_node = random.choice(target_nodes)

        if (target_node.value in gb.UNARY_OPERATORS and are_compatible(target_node.value, source_node.evaluate_tree_from_node())):
            tmp = target_node.right
            target_node.right = source_node
            if validate_after_replacement(target_tree.root, target_node):
                target_tree.update_depths_to_root(target_node)
                return True
            target_node.right = tmp

        elif (target_node.value in gb.BINARY_OPERATORS):
            choice = random.choice(["right", "left"])
            if choice == "right" and are_compatible(target_node.value, source_node.evaluate_tree_from_node(), target_node.left.evaluate_tree_from_node()):
                tmp = target_node.right
                target_node.right = source_node
                if validate_after_replacement(target_tree.root, target_node):
                    target_tree.update_depths_to_root(target_node)
                    return True
                target_node.right = tmp
            elif choice == "left" and are_compatible(target_node.value, target_node.right.evaluate_tree_from_node(), source_node.evaluate_tree_from_node()):
                tmp = target_node.left
                target_node.left = source_node
                if validate_after_replacement(target_tree.root, target_node):
                    target_tree.update_depths_to_root(target_node)
                    return True
                target_node.left = tmp
        target_nodes.remove(target_node)
    return False


def generate_initial_solution(input_variables=None):
    """
    Generate an initial solution for the GP algorithm.

    Args:
        input_variables (list, optional): List of variables to be used in the tree. Defaults to all available variables.

    Returns:
        Tree: The initial solution tree.

    Raises:
        KeyError: If not enough variables are provided.
    """    
    if input_variables is None:
        input_variables = list(gb.VARIABLES_MAP.keys())
    variables = input_variables[:]
    n_variables = len(variables)
    if n_variables != 0:
        n_leaves = int(2 ** np.ceil(np.log2(n_variables)))
        n_actual_leaves = n_leaves * random.choice([2, 4])
        max_depth = np.log2(n_actual_leaves)
    else:
        raise KeyError("Not enough variables in general_initial_solution")

    while True:
        root = random_initial_tree(0, max_depth, variables)
        try:
            if root.validate_tree_from_node():
                root.evaluate_tree_from_node()
                tree = Tree(root, max_depth)
                return tree
            else:
                print("not valid")
        except:
            pass
        variables = input_variables[:]