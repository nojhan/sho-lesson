import numpy as np

from . import pb

########################################################################
# Objective functions
########################################################################

# Decoupled from objective functions, so as to be used in display.
def to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    sensors = []
    for i in range(0,len(sol),2):
        sensors.append( ( int(round(sol[i])), int(round(sol[i+1])) ) )
    return sensors


def cover_sum(sol, domain_width, sensor_range):
    """Compute the coverage quality of the given vector."""
    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    return np.sum(pb.coverage(domain, sensors, sensor_range))


########################################################################
# Initialization
########################################################################

def rand(dim, scale):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim) * scale


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale):
    """Draw a random vector in a square of witdh `scale`
    around the given one."""
    # TODO handle constraints
    new = sol + (np.random.random(len(sol)) * scale - scale/2)
    return new

