import numpy as np
from generators import get_coefficient_ranges, get_constant_ranges
from utils import compute_weights_sim

BINARY_OPERATORS = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    # "^": np.pow
}

#https://numpy.org/doc/2.1/reference/routines.math.html
UNARY_OPERATORS = {
        "": lambda x: x,  
        "sin": np.sin,
        "cos": np.cos,
        "tan":np.tan,
        "log": np.log,
        "arccos": np.arccos,
        "arcsin":np.arcsin,
        "arctan":np.arctan,
        "sqrt":np.sqrt,
        "cbrt":np.cbrt,
        "abs":np.abs,
        "reciprocal":np.reciprocal
        # "exp": np.exp
    }

X, Y, PROBLEM_SIZE, VARIABLES_MAP, COEFFICIENT_RANGES, CONSTANT_RANGES, UNARY_WEIGHTS = None, None, None, None, None, None, None


def initialize_globals_for_problem(problem_id):
    global X, Y, PROBLEM_SIZE, VARIABLES_MAP, COEFFICIENT_RANGES, CONSTANT_RANGES, UNARY_WEIGHTS
    problem = np.load(f"../data/problem_{problem_id}.npz")

    X = problem['x']
    Y = problem['y']

    PROBLEM_SIZE  = np.shape(X)[0]
    VARIABLES_MAP = {f"X_{i}": X[i] for i in range(PROBLEM_SIZE)} # {'X_0': [1, 2, 3], 'X_1': [4, 5, 6], 'X_2': [7, 8, 9]}
    COEFFICIENT_RANGES = get_coefficient_ranges(X,Y)
    CONSTANT_RANGES = get_constant_ranges(Y)
    UNARY_WEIGHTS = compute_weights_sim(list(UNARY_OPERATORS.keys()))
