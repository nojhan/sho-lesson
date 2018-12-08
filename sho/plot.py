import numpy as np
from matplotlib import cm
import itertools

def surface(ax, shape, f):
    Z = np.zeros( shape )
    for y in range(shape[0]):
        for x in range(shape[1]):
            Z[y][x] = f( (x/shape[0],y/shape[1]), 0.5 )

    X = np.arange(0,shape[0],1)
    Y = np.arange(0,shape[1],1)
    X,Y = np.meshgrid(X,Y)
    ax.plot_surface(X, Y, Z, cmap=cm.viridis)

def path(ax, shape, history):
    def pairwise(iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    k=0
    for i,j in pairwise(range(len(history)-1)):
        xi = history[i][1][0]*shape[0]
        yi = history[i][1][1]*shape[1]
        zi = history[i][0]
        xj = history[j][1][0]*shape[0]
        yj = history[j][1][1]*shape[1]
        zj = history[j][0]
        x = [xi, xj]
        y = [yi, yj]
        z = [zi, zj]
        ax.plot(x,y,z, color=cm.RdYlBu(k))
        k+=1

