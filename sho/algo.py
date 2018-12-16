
########################################################################
# Algorithms
########################################################################

def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 0
    while again(i, val, sol):
        sol = init()
        val = func(sol)
        if val > best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        if val > best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol

# TODO add a simulated annealing solver.
# TODO add a population-based stochastic heuristic template.


