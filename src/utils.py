import numpy as np
import globals as gb

MAX_EXP = 709


def are_compatible(operator, right_val, left_val=0):
    """
    Check if two values are compatible with a given operator. This is
    used to validate the syntax of the generated expression.

    Parameters
    ----------
    operator : str
        The operator to check for compatibility.
    right_val : int or float or numpy.ndarray
        The right value of the operator.
    left_val : int or float or numpy.ndarray, optional
        The left value of the operator. Defaults to 0.

    Returns
    -------
    bool
        True if the values are compatible with the operator, False otherwise.
    """
    match operator:
        case "+":
            if not isinstance(right_val, np.ndarray) and not isinstance(left_val, np.ndarray):
                return False if right_val == 0 and left_val == 0 else True
            else: # check if all right_vals are non-zero
                summation = np.add(right_val, left_val)
                if np.all(summation == 0):
                    return False
                return True
        case "-":
            if not isinstance(right_val, np.ndarray) and not isinstance(left_val, np.ndarray):
                return False if right_val == 0 and left_val == 0 else True
            else: # check if all right_vals are non-zero
                difference = np.subtract(right_val, left_val)
                if np.all(difference == 0):
                    return False
                return True
        case "*":
            if not isinstance(right_val, np.ndarray) and not isinstance(left_val, np.ndarray):
                return False if right_val == 0 or left_val == 0 else True               
            else:
                prod = np.multiply(right_val, left_val)
                if np.all(prod == 0):
                    return False
                return True
        case "/":
            if not isinstance(right_val,np.ndarray):
                return False if right_val == 0 else True
            else : # check if all right_vals are non-zero
                return False if (0 in right_val) else True
            
        case "^":
            if not isinstance(right_val, np.ndarray) and not isinstance(left_val, np.ndarray): # (non array) ^ (non array)
                return False if ( left_val < 0 and not int(right_val) == right_val) or (left_val == 0 and right_val < 0) else True
            elif not isinstance(right_val, np.ndarray) and isinstance(left_val, np.ndarray): # array ^ (non array)
                return False if (not int(right_val) == right_val and np.any(left_val < 0)) or (right_val < 0 and np.any(left_val == 0)) else True
            elif isinstance(right_val, np.ndarray) and not isinstance(left_val, np.ndarray): # (non array) ^ array
                return False if (left_val < 0 and any((not int(i) == i) for i in right_val)) or (left_val == 0 and np.any(right_val < 0)) else True
            else : # array ^ array
                for i in range(len(right_val)):                    
                    if (left_val[i] < 0 and not int(right_val[i]) == right_val[i]) or (left_val[i] == 0 and right_val[i] < 0):
                        return False 
                return True
        
        case "exp":
            if not isinstance(right_val,np.ndarray):
                return False if right_val > MAX_EXP else True
            else : # check if all right_vals are less than MAX_EXP
                return False if (np.any(right_val > MAX_EXP)) else True
            
        case "log" :
            if not isinstance(right_val,np.ndarray):
                return False if right_val <= 0 else True
            else : # check if all right_vals are non-negative
                return False if (np.any(right_val <= 0)) else True
        
        case "arccos" :
            if not isinstance(right_val,np.ndarray):
                return False if right_val < -1 or right_val > 1 else True
            else : # check if all right_vals are between -1 and 1
                return False if (np.any(right_val < -1 ) or np.any(right_val > 1) ) else True
            
        case "arcsin" :
            if not isinstance(right_val,np.ndarray):
                return False if right_val < -1 or right_val > 1 else True
            else : # check if all right_vals are between -1 and 1
                return False if (np.any(right_val < -1 )or np.any(right_val > 1) )else True
        
        case "sqrt" :
            if not isinstance(right_val,np.ndarray):
                return False if right_val < 0 else True
            else : # check if all right_vals are non-negative
                return False if (np.any(right_val < 0) )else True

        case "reciprocal" :
            if not isinstance(right_val,np.ndarray):
                return False if right_val == 0 else True
            else:
                return False if (np.any(right_val == 0)) else True
            
        case "tan" :
            if not isinstance(right_val,np.ndarray):
                k = (right_val - np.pi / 2) / np.pi
                return False if k.is_integer() else True
            else:
                for i in range(len(right_val)):
                    k = (right_val[i] - np.pi / 2) / np.pi
                    if k.is_integer():
                        return False
                return True

        case _:
            return True


def mse(y_computed: np.ndarray, y_expected: np.ndarray):
    return 100 * mse2(y_computed, y_expected)


def mse2(y_computed: np.ndarray, y_expected: np.ndarray):
    """
    Computes the mean squared error (MSE) between y_computed and y_expected
    
    Parameters
    ----------
    y_computed : numpy array
        The computed y values
    y_expected : numpy array
        The expected y values
    
    Returns
    -------
    mse : float
        The mean squared error between y_computed and y_expected
    """
    return np.square(y_expected - y_computed).sum() / len(y_expected)


def similarity(operator, x_i, y):
    """
    Computes the similarity between y and y_pred, normalized to [0,1]
    
    Parameters
    ----------
    operator : callable
        The operator to apply to x_i
    x_i : numpy array
        The input to apply the operator to
    y : numpy array
        The expected output
    """    
    y_pred = operator(x_i)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    y_pred_norm = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)
    return 1 / (1 + mse2(y_pred_norm, y_norm))


def compute_weights_sim(operators):
    """
    Computes the weights for the given unary operators based on the similarity between the expected output and the output of the operator when applied to the input.
    
    Parameters
    ----------
    operators : list
        The list of unary operators to compute the weights for
    
    Returns
    -------
    weights : dict
        A dictionary where the keys are the operators and the values are their corresponding weights
    """
    weights = {op: 0 for op in operators}
    for op in weights:
        max_sim = 0
        for i in range(np.shape(gb.X)[0]):
            if (are_compatible(op, gb.X[i])):
                sim = similarity(gb.UNARY_OPERATORS[op], gb.X[i], gb.Y)
                max_sim = max(max_sim, sim)
               
        weights[op] = max_sim    
    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    else:
        # If all correlations failed, use uniform weights
        weights = {k: 1.0/len(operators) for k in operators}
    return weights


def sort_individuals(population, mse_weight=0.6):
    """
    Sorts a population of individuals based on a weighted combination of their MSE and sign scores.

    Parameters
    ----------
    population : list
        A list of individuals, each having a `fitness` attribute that is a tuple consisting of an MSE score and a sign score.
    mse_weight : float, optional
        The weight given to the MSE score in the combined score calculation. Defaults to 0.6.

    Returns
    -------
    list
        The sorted population of individuals in descending order of their combined scores.
    list
        The combined scores of the sorted population.
    """

    mse_scores = np.array([ind.fitness[0] for ind in population])
    sign_scores = np.array([ind.fitness[1] for ind in population])

    # can't sort if all mse's are equal
    diff_max_min = (mse_scores.max() - mse_scores.min())
    if diff_max_min == 0:
        diff_max_min = 1
    # Min-max normalization
    mse_norm = (mse_scores - mse_scores.min()) / diff_max_min 
    sign_norm = sign_scores / 100  # Already in [0,100] range
    
    # Combine scores with weighted sum
    combined_scores = mse_weight * mse_norm + (1 - mse_weight) * sign_norm

    # Sort population left_vald on combined scores (descending order)
    sorted_pairs = sorted(zip(combined_scores, population), key=lambda pair: pair[0], reverse=True)
    sorted_scores, sorted_population = zip(*sorted_pairs)
    
    return list(sorted_population), list(sorted_scores)


def compute_score(individual, population, mse_weight=0.76):
    """
    Computes a score for an individual by normalizing its MSE and sign scores to [0,1] and then combining them with a weighted sum.

    Parameters
    ----------
    individual : Individual
        The individual to compute a score for
    population : list
        The population of individuals to normalize scores relative to
    mse_weight : float, optional
        The weight given to the MSE score in the combined score calculation. Defaults to 0.76.

    Returns
    -------
    float
        The combined score of the individual
    """
    mse_scores = np.array([ind.fitness[0] for ind in population])
    
    # Min-max normalization
    mse_norm = (individual.fitness[0] - mse_scores.min()) / (mse_scores.max() - mse_scores.min())
    sign_norm = individual.fitness[1] / 100  # Already in [0,100] range
    
    # Combine scores with weighted sum
    return mse_weight * mse_norm + (1 - mse_weight) * sign_norm


def get_unary_weights(avaialble_operators):
    """
    Get the weights of the available unary operators.

    Parameters
    ----------
    avaialble_operators : list
        List of available unary operators.

    Returns
    -------
    list
        List of normalized weights of the available unary operators.
    """
    available_weights = [w for op, w in gb.UNARY_WEIGHTS.items() if op in avaialble_operators]

    # normalize weights
    total = sum(available_weights)
    if total > 0:
        available_weights = [w/total for w in available_weights]
    else:
        # If all correlations failed, use uniform weights
        available_weights = [1.0/len(avaialble_operators) for _ in avaialble_operators]
    return available_weights
