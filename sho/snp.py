import sys
import numpy as np
import matplotlib.pyplot as plt
import copy


########################################################################
# Utilities
########################################################################

def x(a):
    """Return the first element of a 2-tuple.
    >>> x([1,2])
    1
    """
    return a[0]


def y(a):
    """Return the second element of a 2-tuple.
    >>> y([1,2])
    2
    """
    return a[1]


def distance(a,b):
    """Euclidean distance (in pixels).

    >>> distance( (1,1),(2,2) ) == math.sqrt(2)
    True
    """
    return np.sqrt( (x(a)-x(b))**2 + (y(a)-y(b))**2 )


def highlight_sensors(domain, sensors, val=2):
    """Add twos to the given domain, in the cells where the given
    sensors are located.

    >>> highlight_sensors( [[0,0],[1,1]], [(0,0),(1,1)] )
    [[2, 0], [1, 2]]
    """
    for s in sensors:
        # `coverage` fills the domain with ones,
        # adding twos will be visible in an image.
        domain[y(s)][x(s)] = val
    return domain


########################################################################
# Objective functions
########################################################################

def coverage(domain, sensors, sensor_range):
    """Set a given domain's cells to on if they are visible
    from one of the given sensors at the given sensor_range.

    >>> snp.coverage(np.zeros((5,5)),[(2,2)],2)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """
    for py in range(len(domain)):
        for px in range(len(domain[py])):
            p = (px,py)
            for x in sensors:
                if distance(x,p) < sensor_range:
                    domain[py][px] = 1
                    break
    return domain


# Decoupled from objective functions, so as to be used in display.
def num_to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> num_to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    sensors = []
    for i in range(0,len(sol),2):
        sensors.append( ( int(round(sol[i])), int(round(sol[i+1])) ) )
    return sensors


def bit_to_sensors(sol):
    """Convert an square array of d lines/columns containing n ones
    to an array of n 2-tuples with related coordinates.

    >>> bit_to_sensors([[1,0],[1,0]])
    [(0, 0), (0, 1)]
    """
    sensors = []
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            if sol[i][j] == 1:
                sensors.append( (j,i) )
    return sensors


def bit_cover_sum(sol, domain_width, sensor_range):
    """Compute the coverage quality of the given array of bits."""
    domain = np.zeros((domain_width,domain_width))
    sensors = bit_to_sensors(sol)
    return np.sum(coverage(domain, sensors, sensor_range))


def num_cover_sum(sol, domain_width, sensor_range):
    """Compute the coverage quality of the given vector."""
    domain = np.zeros((domain_width,domain_width))
    sensors = num_to_sensors(sol)
    return np.sum(coverage(domain, sensors, sensor_range))


def make_func(cover, **kwargs):
    """Make an objective function from the given function.
    An objective function takes a solution and returns a scalar."""
    def f(sol):
        return cover(sol,**kwargs)
    return f


########################################################################
# Initialization
########################################################################

def num_rand(dim, scale):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim) * scale


def bit_rand(domain_width, nb_sensors):
    """"Draw a random domain containing nb_sensors ones."""
    domain = np.zeros( (domain_width,domain_width) )
    for x,y in np.random.randint(0, domain_width, (nb_sensors, 2)):
        domain[y][x] = 1
    return domain


def make_init(init, **kwargs):
    """Make an initialization operator from the given function.
    An init. op. returns a solution."""
    def f():
        return init(**kwargs)
    return f


########################################################################
# Neighborhood
########################################################################

def num_neighb_square(sol, scale):
    """Draw a random vector in a square of witdh `scale`
    around the given one."""
    # TODO handle constraints
    new = sol + (np.random.random(len(sol)) * scale - scale/2)
    return new


def bit_neighb_square(sol, scale):
    """Draw a random array by moving ones to adjacent cells."""
    # Copy, because Python pass by reference
    # and we may not the to alter the original solution.
    new = copy.copy(sol)
    for py in range(len(sol)):
        for px in range(len(sol[py])):
            if sol[py][px] == 1:
                new[py][px] = 0 # Remove original position.
                # TODO handle constraints
                d = np.random.randint(-scale//2,scale//2,2)
                new[py+y(d)][px+x(d)] = 1
    return new


def make_neig(neighb, **kwargs):
    """Make an neighborhood operator from the given function.
    A neighb. op. takes a solution and returns another one."""
    def f(sol):
        return neighb(sol, **kwargs)
    return f


########################################################################
# Stopping criterions
########################################################################

def iter_max(i, val, sol, nb_it):
    if i < nb_it:
        return True
    else:
        return False

def make_iter(iters, **kwargs):
    """Make an iterations operator from the given function.
    A iter. op. takes a value and a solution and returns
    the current number of iterations."""
    def f(i, val, sol):
        return iters(i, val, sol, **kwargs)
    return f


# Stopping criterions that are actually just checkpoints.

def combine(i, val, sol, agains):
    """Combine  several stopping criterions in one."""
    res = True
    for again in agains:
        res = res and again(i, val, sol)
    return res


def save(i, val, sol, filename="run.csv", fmt="{it} ; {val} ; {sol}\n"):
    """Save all iterations to a file."""
    # Append a line at the end of the file.
    with open(filename.format(it=i), 'a') as fd:
        fd.write( fmt.format(it=i, val=val, sol=sol) )
    return True # No incidence on termination.


def iter_log(i, val, sol, fmt="{it} {val}\n"):
    """Print progress on stderr."""
    sys.stderr.write( fmt.format(it=i, val=val) )
    return True


########################################################################
# Algorithms
########################################################################

def random(func, init, again):
    """Iterative random search template."""
    best_sol = None
    best_val = - np.inf
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
    while again(i, val, sol):
        sol = neighb(best_sol)
        val = func(sol)
        if val > best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol

# TODO add a simulated annealing solver.
# TODO add a population-based stochastic heuristic template.


########################################################################
# Interface
########################################################################

if __name__=="__main__":
    import argparse

    # Dimension of the search space.
    d = 2

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=3, type=int,
            help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
            help="Sensors' range (as a fraction of domain width)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=100, type=int,
            help="Domain width (a number of cells)")

    can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
            help="Maximum number of iterations")

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_greedy","bit_greedy"]
    can.add_argument("-m", "--solver", metavar="NAME", choices=solvers, default="num_greedy",
            help="Solver to use, among: "+", ".join(solvers))

    # TODO add the corresponding stopping criterion.
    can.add_argument("-t", "--target", metavar="VAL", default=1e-3, type=float,
            help="Function value target delta")

    the = can.parse_args()

    # Minimum checks.
    assert(0 < the.nb_sensors)
    assert(0 < the.sensor_range <= 1)
    assert(0 < the.domain_width)
    assert(0 < the.iters)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(the.seed)

    # Weird numpy way to ensure single line print of array.
    np.set_printoptions(linewidth = np.inf)

    domain = np.zeros((the.domain_width, the.domain_width))

    # Common termination and checkpointing.
    iters = make_iter(
                combine,
                agains = [
                    make_iter(iter_max,
                        nb_it = the.iters),
                    make_iter(save,
                        filename = the.solver+".csv",
                        fmt = "{it} ; {val} ; {sol}\n"),
                    make_iter(iter_log,
                        fmt="\r{it} {val}")
                ]
            )

    # Erase the previous file.
    with open(the.solver+".csv", 'w') as fd:
        fd.write("# {} {}\n".format(the.solver,the.domain_width))

    if the.solver == "num_greedy":
        val,sol = greedy(
                make_func(num_cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make_init(num_rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make_neig(num_neighb_square,
                    scale = the.domain_width/10), # TODO think of an alternative.
                iters
            )
        sensors = num_to_sensors(sol)

    elif the.solver == "bit_greedy":
        val,sol = greedy(
                make_func(bit_cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range),
                make_init(bit_rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make_neig(bit_neighb_square,
                    scale = the.domain_width/10),
                iters
            )
        sensors = bit_to_sensors(sol)


    # Fancy output.
    print("\n",val,":",sensors)

    domain = coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = highlight_sensors(domain, sensors)
    plt.imshow(domain)
    plt.show()

