import numpy as np

def sphere(x,offset=0.5):
    """Computes the square of a multi-dimensional vector x."""
    f = 0
    for i in range(len(x)):
        f += (x[i]-offset)**2
    return f

def square(sol,scale=1):
    """Gnerate a random vector close at thegiven scale to the given sol."""
    return sol + np.random.random(len(sol))*scale

def greedy(objective_function, dimension, iterations, target=1e-3, neighborhood=square, scale=1/100, history=None):
    """Search the given objective_function of the given dimension,
    during the given number of iterations, generating solution
    with the given neighborhood.
    Returns the best value of the function and the best solution."""
    best_sol = np.random.random(dimension)
    best_val = objective_function(best_sol)
    for i in range(iterations):
        sol = neighborhood(best_sol,scale)
        val = objective_function(sol)
        if val < best_val:
            best_val = val
            best_sol = sol
        if history is not None:
            history.append((val,sol))
        if val < target: # Assume the optimum is zero
            break
    return best_val, best_sol


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import argparse
    import plot

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dim", metavar="NB", default=2, type=int,
            help="Number of dimensions")

    functions = {"sphere":sphere}
    parser.add_argument("-f", "--func", metavar="NAME", choices=functions, default="sphere",
            help="Objective function")

    parser.add_argument("-i", "--iter", metavar="NB", default=1000, type=int,
            help="Maximum number of iterations")

    parser.add_argument("-t", "--target", metavar="VAL", default=1e-3, type=float,
            help="Function value target delta")

    parser.add_argument("-s", "--seed", metavar="VAL", default=0, type=int,
            help="Random pseudo-generator seed (0 for epoch)")

    asked = parser.parse_args()

    np.random.seed(asked.seed)

    history = []
    val,sol = greedy(functions[asked.func], asked.dim, asked.iter, asked.target, square, 0.03, history)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    shape = (20,20)
    plot.surface(ax, shape, sphere)
    plot.path(ax, shape, history)
    plt.show()
