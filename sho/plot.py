import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D

from . import x,y,distance

def sphere(x,offset=0.5):
    """Computes the square of a multi-dimensional vector x."""
    f = 0
    for i in range(len(x)):
        f += (x[i]-offset)**2
    return -1 * f


def surface(ax, shape, f):
    Z = np.zeros( shape )
    for y in range(shape[0]):
        for x in range(shape[1]):
            Z[y][x] = f( (x,y), shape[0]/2 )

    X = np.arange(0,shape[0],1)
    Y = np.arange(0,shape[1],1)
    X,Y = np.meshgrid(X,Y)
    #ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax.plot_surface(X, Y, Z)


def path(ax, shape, history):
    def pairwise(iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    k=0
    for i,j in pairwise(range(len(history)-1)):
        xi = history[i][1][0]
        yi = history[i][1][1]
        zi = history[i][0]
        xj = history[j][1][0]
        yj = history[j][1][1]
        zj = history[j][0]
        x = [xi, xj]
        y = [yi, yj]
        z = [zi, zj]
        ax.plot(x,y,z, color=cm.RdYlBu(k))
        k+=1


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


if __name__=="__main__":
    import snp

    w = 100
    shape = (w,w)
    history = []

    val,sol = snp.greedy(
            snp.make_func(sphere,
                offset = w/2),
            snp.make_init(snp.num_rand,
                dim = 2 * 1,
                scale = w),
            snp.make_neig(snp.num_neighb_square,
                scale = w/10),
            snp.make_iter(
                    snp.several,
                    agains = [
                        snp.make_iter(snp.iter_max,
                            nb_it = 100),
                        snp.make_iter(snp.history,
                            history = history)
                    ]
                )
        )
    sensors = snp.num_to_sensors(sol)

    #print("\n".join([str(i) for i in history]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surface(ax, shape, sphere)
    path(ax, shape, history)
    plt.show()
