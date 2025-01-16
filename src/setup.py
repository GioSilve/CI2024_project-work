import numpy as np


def read_problem(problem_id):
    problem = np.load('../data/problem_{}.npz'.format(problem_id))

    x = problem['x']
    y = problem['y']

    PROBLEM_SIZE  = np.shape(x)[0]
    # print(PROBLEM_SIZE)

    # print(np.shape(x))

    return x, y, PROBLEM_SIZE


# if __name__ == "__main__":
#     x = read_problem(1)[0]
#     y = read_problem(1)[1]
#     size = read_problem(1)[2]
#     print(x)
#     print(y)
#     print(size)

