import matplotlib.pyplot as plt
import numpy as np

def x(p):
    return p[0]

def y(p):
    return p[1]

def distance(a,b):
    return np.sqrt( (x(a)-x(b))**2 + (y(a)-y(b))**2 )

def count(domain_shape, sensors_positions, radius, output_domain = None):
    s = 0
    Y,X = domain_shape
    for y in range(Y):
        for x in range(X):
            p = (x,y)
            t = 0
            for sensor in sensors_positions:
                if distance( p, sensor ) < radius:
                    t += 1
                    # break
            if output_domain is not None:
                output_domain[y][x] = t
            s += t
    return s

if __name__=="__main__":

    domain = np.zeros( (100,100) )

    sensors = np.round(np.random.random( (3,2) ) * 100)

    s = count(domain.shape, sensors, 40, domain)

    plt.imshow(domain)
    plt.show()
