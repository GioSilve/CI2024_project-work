import numpy as np
import random
import globals as gb

"""functions for generating multiplication coefficients for variables"""
"""------------------------------------------------------------------"""
def optimize_initial_coefficients(x, y):
    """
    Compute basic scaling coefficients to match the range of y for each feature in x.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples, )
        Target values
    
    Returns
    -------
    coefficients : array of shape (n_features, )
        Scaling coefficients for each feature
    """
    y_std = np.std(y)
    
    # For each feature in x
    coefficients = []
    for i in range(x.shape[0]):
        x_i = x[i]
        x_mean = np.mean(x_i)
        x_std = np.std(x_i)
        
        # Basic scaling coefficient to match ranges
        coeff = y_std / (x_std + 1e-8)  # avoid division by zero
        
        coefficients.append(coeff)
    
    return np.array(coefficients)



def get_coefficient_ranges(x,y):
    """
    Compute a range of possible coefficients for each feature in x to approximate the range of y.
    
    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Input features
    y : array-like of shape (n_samples, )
        Target values
    
    Returns
    -------
    ranges : list of tuples
        A list of tuples, where each tuple contains the min and max values of a range of coefficients
        for each feature in x.
    """
    base_coeffs = optimize_initial_coefficients(x, y)
    
    # Create ranges around these coefficients
    ranges = []
    for coeff in base_coeffs:
        # Create range from 0.1× to 10× the optimized coefficient
        min_coeff = coeff * 0.01
        max_coeff = coeff * 5
        ranges.append((min_coeff, max_coeff))
        
    return ranges

def compute_coefficient(x_i, variables_map, ranges):
    """
    Compute a random coefficient for feature x_i in the given range.
    
    Parameters
    ----------
    x_i : string
        The name of the feature
    variables_map : dict
        A dictionary mapping feature names to their corresponding values
    ranges : list of tuples
        A list of tuples, where each tuple contains the min and max values of a range of coefficients
        for each feature in x.
    
    Returns
    -------
    coeff : float
        A random coefficient for feature x_i in the given range.
    """
    x_array = list(variables_map.keys()) 
    index = x_array.index(x_i)
    sign = random.choice([-1, 1])
    return np.random.uniform(ranges[index][0], ranges[index][1])*sign 



"""------------------------------------------------------------------"""
"""functions for generating constant values"""
def get_constant_ranges(y):
    """
    Compute different ranges of constant values based on the given output y.
    
    Parameters
    ----------
    y : array-like of shape (n_samples, )
        The output values
        
    Returns
    -------
    ranges : dict
        A dictionary containing the ranges of constant values for different operations.
        The keys are the operation names ('add_sub', 'mult_div', 'small', 'powers'),
        and the values are tuples of the min and max values of the range.
    """
    
    """"""
    y_std = np.std(y)
    y_mean = np.mean(np.abs(y))
    x_mean = np.mean(np.abs(gb.X))
    
    # Calculate multiplication scaling factor based on input/output ratio
    mult_scale = y_mean / x_mean if x_mean != 0 else y_mean
    
    # Adjust the range based on the scale difference
    mult_range = max(2.0, mult_scale)
    
    # Create different ranges based on operation context
    ranges = {
        'add_sub': (-y_std, y_std),  # for addition/subtraction
        'mult_div': (-mult_range, mult_range),     # for multiplication/division
        'small': (-1.0, 1.0),        # for fine-tuning
        'powers': range(-3, 3)        # for exponents, usually small integers
    }
    return ranges

def generate_constant(operation_type, unary_operators, y):
    """
    Generate a constant value based on the given operation type and output values.
    
    Parameters
    ----------
    operation_type : str
        The type of operation ('+', '-', '*', '/')
    unary_operators : dict
        A dictionary containing the unary operators and their corresponding functions
    y : array-like of shape (n_samples, )
        The output values
        
    Returns
    -------
    constant : float
        A constant value based on the given operation type and output values
    """
    if operation_type in unary_operators.keys():
        return generate_safe_constant(y)    
    ranges = get_constant_ranges(y)    
    if random.random() < 0.25:
        operation_type = 'small'
    else:
        if operation_type == '+' or operation_type == '-':
            operation_type = 'add_sub'
        elif operation_type == '*' or operation_type == '/':
            operation_type = 'mult_div'
        else:  # Integer constants for exponents
            return np.random.choice(ranges['powers'])
        
    min_val, max_val = ranges[operation_type]
    # Float constants for other operations
    return np.random.uniform(min_val, max_val)

def generate_safe_constant(y):
    """
    Generate a constant value that is safe to use with any operation (i.e. not too large, not zero).
    
    Parameters
    ----------
    y : array-like of shape (n_samples, )
        The output values
        
    Returns
    -------
    constant : float
        A constant value that is safe to use with any operation
    """
    y_std = np.std(y)
    
    # Start with a conservative range
    min_val = 0.1  # Avoid zero for division
    max_val = min(2.0, y_std)  # Keep it reasonable for both mult and add
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return np.random.uniform(min_val, max_val)