import numpy as np
import random
import copy
import globals as gb
from tree import Tree, try_swap, validate_after_replacement, get_random_leaf, generate_initial_solution
from generators import compute_coefficient, generate_constant
from utils import are_compatible, get_unary_weights

def subtree_mutation(target_tree: Tree):
    # generate a random source tree to substitute into target_tree

    num_variables = np.random.randint(1, len(gb.VARIABLES_MAP.keys())) if len(gb.VARIABLES_MAP.keys()) > 1 else 1  # take a random number of variables
    variables = random.sample(list(gb.VARIABLES_MAP.keys()), num_variables)

    source_tree_root = generate_initial_solution(variables).root

    return try_swap(source_tree_root, target_tree, True)


def point_mutation(target_tree: Tree):
    nodes = target_tree.get_nodes()

    while nodes:
        node = random.choice(nodes)
        unary_operators = [op for op in  list(gb.UNARY_OPERATORS.keys()) if op != node.value]
        unary_weights = get_unary_weights(unary_operators)
        binary_operators = [op for op in list(gb.BINARY_OPERATORS.keys()) if op != node.value]
        # mutate unary operator with another one
        if (node.value in gb.UNARY_OPERATORS):
            while unary_operators:
                tmp = node.value
                node.value = np.random.choice(unary_operators, p=unary_weights)
                if are_compatible(node.value, node.right.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, node):
                    return True
                
                unary_operators.remove(node.value)
                unary_weights = get_unary_weights(unary_operators)
                node.value = tmp

        # mutate binary operator with another one
        elif (node.value in gb.BINARY_OPERATORS):
            while binary_operators:
                tmp = node.value
                node.value = random.choice(binary_operators)
                if are_compatible(node.value, node.right.evaluate_tree_from_node(), node.left.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, node):
                    return True
                binary_operators.remove(node.value)
                node.value = tmp

        # mutate variable
        elif node.value in gb.VARIABLES_MAP:
            # change the coefficient
            tmp_c = node.coefficient
            tmp_v = node.value
            node.coefficient = compute_coefficient(node.value, gb.VARIABLES_MAP, gb.COEFFICIENT_RANGES)
            node.value = random.choice(list(gb.VARIABLES_MAP))
            parent = target_tree.get_parent_from_root(node)
            if (parent.value in unary_operators and are_compatible(parent.value, parent.right.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, parent)) \
            or (parent.value in binary_operators and are_compatible(parent.value, parent.right.evaluate_tree_from_node(), parent.left.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, parent)):
                return True
            node.coefficient = tmp_c
            node.value = tmp_v
            
        # mutate constant value
        else :
            # change the constant
            tmp = node.value
            parent = target_tree.get_parent_from_root(node)
            node.value = generate_constant(parent.value, gb.UNARY_OPERATORS, gb.Y)
            if (parent.value in unary_operators and are_compatible(parent.value, parent.right.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, parent)) \
            or (parent.value in binary_operators and are_compatible(parent.value, parent.right.evaluate_tree_from_node(), parent.left.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, parent)):
                return True
            node.value = tmp
        
        nodes.remove(node)
    return False
    

def permutation_mutation(target_tree: Tree):
    available_nodes = [node for node in target_tree.get_non_leaves_nodes() if node.value in gb.BINARY_OPERATORS]
    
    while available_nodes:  
        target_node = random.choice(available_nodes)
        tmpr = target_node.right
        tmpl = target_node.left
        target_node.right = tmpl
        target_node.left = tmpr
        if are_compatible(target_node.value, target_node.right.evaluate_tree_from_node(), target_node.left.evaluate_tree_from_node()) and validate_after_replacement(target_tree.root, target_node):
            return True
        target_node.right = tmpr
        target_node.left = tmpl
        available_nodes.remove(target_node)
    return False


def expansion_mutation(target_tree: Tree):
    leaves = target_tree.get_leaves_nodes()
    binary_operators = list(gb.BINARY_OPERATORS.keys())
    unary_operators = list(gb.UNARY_OPERATORS.keys())
    num_variables = np.random.randint(1, len(gb.VARIABLES_MAP.keys())) if len(gb.VARIABLES_MAP.keys()) > 1 else 1  # take a random number of variables
    variables = random.sample(list(gb.VARIABLES_MAP.keys()), num_variables)

    while leaves:
        target_node = random.choice(leaves)
        source_tree_root = generate_initial_solution(variables).root
        parent = target_tree.get_parent_from_root(target_node)

        if (parent.value in unary_operators and are_compatible(parent.value, source_tree_root.evaluate_tree_from_node())):
            tmp = parent.right
            parent.right = source_tree_root
            if validate_after_replacement(target_tree.root, parent):
                target_tree.update_depths_to_root(parent)
                return True
            parent.right = tmp

        elif (parent.value in binary_operators):
            choice = random.choice(["right", "left"])
            if choice == "right" and are_compatible(parent.value, source_tree_root.evaluate_tree_from_node(), parent.left.evaluate_tree_from_node()):
                tmp = parent.right
                parent.right = source_tree_root
                if validate_after_replacement(target_tree.root, parent):
                    target_tree.update_depths_to_root(parent)
                    return True
                parent.right = tmp
            elif choice == "left" and are_compatible(parent.value, parent.right.evaluate_tree_from_node(), source_tree_root.evaluate_tree_from_node()):
                tmp = parent.left
                parent.left = source_tree_root
                if validate_after_replacement(target_tree.root, parent):
                    target_tree.update_depths_to_root(parent)
                    return True
                parent.left = tmp
        leaves.remove(target_node)
    return False


def hoist_mutation(target_tree: Tree):
    # substitute the root with a subtree
    nodes = target_tree.get_non_leaves_nodes()
    root = target_tree.root

    if len(nodes) == 1:
        return False
    
    #remove root
    nodes.remove(root)

    node = random.choice(nodes)

    target_tree.root = node
    return True


def collapse_mutation(target_tree: Tree):
    available_nodes = target_tree.get_non_leaves_nodes()

    # Consider only nodes whose children are not leaves
    available_nodes = [node for node in available_nodes if node.right.right or node.right.left]
    
    unary_operators = list(gb.UNARY_OPERATORS.keys())
    binary_operators = list(gb.BINARY_OPERATORS.keys())
    
    max_attempts = 10
    while available_nodes:
        parent = random.choice(available_nodes)
        attempts = 0
        while attempts < max_attempts:
            if random.choice([0, 1]): # create a radnom leaf
                random_leaf = get_random_leaf()
            else:
                random_leaf = copy.deepcopy(random.choice(parent.get_leaves_nodes_from_node()))

            if (parent.value in unary_operators and are_compatible(parent.value, random_leaf.evaluate_tree_from_node())):
                tmp = parent.right
                parent.right = random_leaf
                if validate_after_replacement(target_tree.root, parent):
                    target_tree.update_depths_to_root(parent)
                    return True
                parent.right = tmp

            elif (parent.value in binary_operators):
                choice = random.choice(["right", "left"])
                if choice == "right" and are_compatible(parent.value, random_leaf.evaluate_tree_from_node(), parent.left.evaluate_tree_from_node()):
                    tmp = parent.right
                    parent.right = random_leaf
                    if validate_after_replacement(target_tree.root, parent):
                        target_tree.update_depths_to_root(parent)
                        return True
                    parent.right = tmp
                elif choice == "left" and are_compatible(parent.value, parent.right.evaluate_tree_from_node(), random_leaf.evaluate_tree_from_node()):
                    tmp = parent.left
                    parent.left = random_leaf
                    if validate_after_replacement(target_tree.root, parent):
                        target_tree.update_depths_to_root(parent)
                        return True
                    parent.left = tmp
            attempts += 1
        available_nodes.remove(parent)
    return False


MUTATIONS = {
    "subtree": subtree_mutation,
    "point": point_mutation,
    "permutation": permutation_mutation,
   "hoist": hoist_mutation,
    "expansion": expansion_mutation,
    "collapse": collapse_mutation
}

MUTATIONS_WEIGHTS = {
    "subtree": 0.25 , #0.25
    "point": 0.3 , #0.3
    "permutation": 0.1, #0.1
    "hoist": 0.1, #0.1
    "expansion": 0.15, #0.15
    "collapse": 0.1 #0.1
 }


def scale_weights_by_depth(weights, depth):
    n_expected_leaves = int(2 ** np.ceil(np.log2(gb.PROBLEM_SIZE)))
    expected_depth = np.log2(n_expected_leaves * 2) + 1
    if depth > expected_depth and depth < expected_depth + 2 :
        for m in weights.keys():
            if m == "hoist" or m == "collapse":
                weights[m] = 0.2
            elif m == "expansion":
                weights[m] = 0.1
            elif m == "point":
                weights[m] = 0.25
            elif m == "permutation":
                weights[m] = 0.1
            elif m == "subtree":
                weights[m] = 0.15
                
    elif depth >= expected_depth + 2 and depth <= expected_depth + 4 :
        for m in weights.keys():
            if m == "hoist" or m == "collapse":
                weights[m] = 0.3
            elif m == "expansion":
                weights[m] = 0.05
            elif m == "point":
                weights[m] = 0.2
            elif m == "permutation":
                weights[m] = 0.1
            elif m == "subtree":
                weights[m] = 0.05
    elif depth > 10:
        for m in weights.keys():
            if m == "hoist" or m == "collapse":
                weights[m] = 0.5
            else:
                weights[m] = 0
          
    return list(weights.values())


def mutation(genome: Tree):
    """
    Perform a mutation on the given tree genome based on available mutation types.
    
    This function selects a mutation type from the defined MUTATIONS based on 
    their scaled weights relative to the depth of the tree. It tries to apply
    the selected mutation to the genome. If successful, it returns True. If 
    not, it removes that mutation type from the list and rescales the weights, 
    continuing the process until all mutation types are exhausted. If no mutation 
    is successful, it returns False.

    Parameters:
    genome (Tree): The tree structure representing an individual solution that 
                   is subject to mutation.

    Returns:
    bool: True if a mutation was successfully applied; False otherwise.
    """

    available_mutations = list(MUTATIONS.keys())
    available_mutations_weights = {m:p for m,p in MUTATIONS_WEIGHTS.items()}

    w_mutations = scale_weights_by_depth(available_mutations_weights, genome.depth)

    while available_mutations:
        mutation = np.random.choice(available_mutations, p=w_mutations)
        # print(mutation)
        if MUTATIONS[mutation](genome):
            return True
        # scale weights 
        to_remove = w_mutations[available_mutations.index(mutation)]
        w_mutations.remove(to_remove)

        # recompute weights
        w_mutations = [w / sum(w_mutations) for w in w_mutations]

        available_mutations.remove(mutation)
    return False