from . import distance

########################################################################
# Objective functions
########################################################################

def coverage(domain, sensors, sensor_range):
    """Set a given domain's cells to on if they are visible
    from one of the given sensors at the given sensor_range.

    >>> coverage(np.zeros((5,5)),[(2,2)],2)
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


