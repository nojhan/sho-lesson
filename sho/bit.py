import numpy as np
import copy

from . import x,y,pb

########################################################################
# Objective functions
########################################################################

def cover_sum(sol, domain_width, sensor_range):
    """Compute the coverage quality of the given array of bits."""
    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    return np.sum(pb.coverage(domain, sensors, sensor_range))


def to_sensors(sol):
    """Convert an square array of d lines/columns containing n ones
    to an array of n 2-tuples with related coordinates.

    >>> to_sensors([[1,0],[1,0]])
    [(0, 0), (0, 1)]
    """
    sensors = []
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            if sol[i][j] == 1:
                sensors.append( (j,i) )
    return sensors


########################################################################
# Initialization
########################################################################

def rand(domain_width, nb_sensors):
    """"Draw a random domain containing nb_sensors ones."""
    domain = np.zeros( (domain_width,domain_width) )
    for x,y in np.random.randint(0, domain_width, (nb_sensors, 2)):
        domain[y][x] = 1
    return domain


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale):
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

