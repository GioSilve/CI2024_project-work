import numpy as np
import globals as gb

MAX_EXP = 709

def are_compatible(operator, right_val, left_val=0):
    # print("are compatible")
    # print(f"operator: {operator} - right val: {right_val} - left val: {left_val}")
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
    # 100*np.square(y_train-d3584.f(x_train)).sum()/len(y_train):g}")
    return 100 * np.square(y_expected - y_computed).sum() / len(y_expected)

def mse2(y_computed: np.ndarray, y_expected: np.ndarray):
    return np.square(y_expected - y_computed).sum() / len(y_expected)

def similarity(operator, x_i, y):
    # Add normalization to make comparison fairer across different scales
    y_pred = operator(x_i)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    y_pred_norm = (y_pred - y_pred.mean()) / (y_pred.std() + 1e-8)
    return 1 / (1 + mse2(y_pred_norm, y_norm))

def compute_weights_sim(operators):
    weights = {op: 0 for op in operators}
    # print(f"operators {operators}")
    # print(f"weights {weights}")
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
    # print(f"weights {weights}")
    return weights

def sort_individuals(population, mse_weight=0.6):
    # Normalize both components to [0,1] scale across the population
    mse_scores = np.array([ind.fitness[0] for ind in population])
    sign_scores = np.array([ind.fitness[1] for ind in population])
    if not population:
        print("ENSOMMA")
    # can't sort if all mse's are equal
    diff_max_min = (mse_scores.max() - mse_scores.min())
    if diff_max_min == 0 :
        diff_max_min = 1
    # Min-max normalization
    mse_norm = (mse_scores - mse_scores.min()) / diff_max_min 
    sign_norm = sign_scores / 100  # Already in [0,100] range
    
    # Combine scores with weighted sum
    combined_scores = mse_weight * mse_norm + (1 - mse_weight) * sign_norm
    
    # sorted_population = [x for _, x in sorted(zip(combined_scores, population), 
    #                                         key=lambda pair: pair[0], 
    #                                         reverse=True)]

    # Sort population left_vald on combined scores (descending order)
    sorted_pairs = sorted(zip(combined_scores, population), key=lambda pair: pair[0], reverse=True)
    sorted_scores, sorted_population = zip(*sorted_pairs)
    
    return list(sorted_population), list(sorted_scores)

def compute_score(individual, population, mse_weight=0.6):
     # Normalize both components to [0,1] scale across the population
    mse_scores = np.array([ind.fitness[0] for ind in population])
    sign_scores = np.array([ind.fitness[1] for ind in population])
    
    # Min-max normalization
    mse_norm = (individual.fitness[0] - mse_scores.min()) / (mse_scores.max() - mse_scores.min())
    sign_norm = individual.fitness[1] / 100  # Already in [0,100] range
    
    # Combine scores with weighted sum
    return mse_weight * mse_norm + (1 - mse_weight) * sign_norm
