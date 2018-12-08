import numpy as np

def sphere(x,offset=0.5):
    """Computes the square of a multi-dimensional vector x."""
    f = 0
    for i in range(len(x)):
        f += (x[i]-offset)**2
    return f

def onemax(x):
    """Sum the given bitstring."""
    s = 0
    for i in x:
        s += i
    return s

def numerical_random(d):
    """Draw a random multi-dimensional vector in [0,1]**d"""
    return np.random.random(d)

def bitstring_random(d):
    """Draw a random bistring of size d, with P(1)=0.5."""
    return [int(round(i)) for i in np.random.random(d)]

def search(objective_function, dimension, iterations, generator, history=None):
    """Search the given objective_function of the given dimension,
    during the given number of iterations, generating random solution
    with the given generator.
    Returns the best value of the function and the best solution."""
    best_val = float("inf")
    best_sol = None
    for i in range(iterations):
        sol = generator(dimension)
        val = objective_function(sol)
        if val < best_val:
            best_val = val
            best_sol = sol
        if history is not None:
            history.append((val,sol))
    return best_val, best_sol


if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import plot

    print("Random search over 10-OneMax")
    print("After 10 iterations:")
    val,sol = search(onemax, 10, 10, bitstring_random)
    print("\t",val,sol)
    print("After 1000 iterations:")
    val,sol = search(onemax, 10, 1000, bitstring_random)
    print("\t",val,sol)

    print("Random search over 2-Sphere")
    print("After 10 iterations:")
    val,sol = search(sphere, 2, 10, numerical_random)
    print("\t",val,sol)
    print("After 50 iterations:")
    history = []
    val,sol = search(sphere, 2, 50, numerical_random, history)
    print("\t",val,sol)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    shape = (20,20)
    plot.surface(ax, shape, sphere)
    plot.path(ax, shape, history)
    plt.show()
