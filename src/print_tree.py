import globals as gb

def print_tree(node, is_root: bool = True):
    """
    Prints the symbolic representation of the tree with the names of the variables
    
    Args:
        node: TreeNode
        is_root: bool
    """
    if not node:
        return

    # Add parentheses around subexpressions unless it's the root
    if not is_root:
        print("(", end="")

    # Traverse the left child
    if node.left:
        print_tree(node.left, is_root=False)

    # Print the current node's value
    print(f"{node.coefficient} {node.value}" if node.coefficient and node.value in gb.VARIABLES_MAP else node.value, end=" ")

    # Traverse the right child
    if node.right:
        print_tree(node.right, is_root=False)

    # Close parentheses if not the root
    if not is_root:
        print(")", end="") 


def print_tree_values(node, is_root: bool = True):
    """ 
    Prints the symbolic representation of the tree with the values of the variables
    
    Args:
        node: TreeNode
        is_root: bool
    """
    if not node:    
        return

    # Add parentheses around subexpressions unless it's the root
    if not is_root:
        print("(", end="")

    # Traverse the left child
    if node.left:
        print_tree_values(node.left, is_root=False)

    # Print the current node's value
    print(f"{node.coefficient} * {gb.VARIABLES_MAP[node.value]}" if node.coefficient and node.value in gb.VARIABLES_MAP else node.value, end=" ")

    # Traverse the right child
    if node.right:
        print_tree_values(node.right, is_root=False)

    # Close parentheses if not the root
    if not is_root:
        print(")", end="") 


def print_expr(node):
    """
    Prints the symbolic representation of the tree with the names of the variables inside an expression
    """
    print_tree(node) 
    print(" = y")

def print_expr_values(node):
    """
    Prints the symbolic representation of the tree with the values of the variables inside an expression
    """
    print_tree_values(node) 
    print(" = ", end="")
    print(node.evaluate_tree_from_node())