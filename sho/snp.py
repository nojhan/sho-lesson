import numpy as np
import matplotlib.pyplot as plt
import copy

########################################################################
# Utilities
########################################################################

def x(a):
    return a[0]

def y(a):
    return a[1]

def distance(a,b):
    return np.sqrt( (x(a)-x(b))**2 + (y(a)-y(b))**2 )


########################################################################
# Objective functions
########################################################################

def coverage(domain,sensors,sensor_range):
    for py in range(len(domain)):
        for px in range(len(domain[py])):
            p = (px,py)
            for x in sensors:
                if distance(x,p) < sensor_range:
                    domain[py][px] = 1
                    break
    return domain

def cover_bit(sol,domain_width,sensor_range):
    domain = np.zeros((domain_width,domain_width))
    sensors = []
    for i in range(domain_width):
        for j in range(domain_width):
            if sol[i][j] == 1:
                sensors.append( (j,i) )
    return np.sum(coverage(domain, sensors, sensor_range))

def cover_num(sol,domain_width,sensor_range):
    domain = np.zeros((domain_width,domain_width))
    sensors = []
    for i in range(0,len(sol),2):
        sensors.append( (sol[i],sol[i+1]) )
    return np.sum(coverage(domain, sensors, sensor_range))

def make_func(cover,**kwargs):
    def f(sol):
        return cover(sol,**kwargs)
    return f


########################################################################
# Initialization
########################################################################

def rand_num(dim):
    return np.random.random(dim)

def rand_bit(domain_width,nb_sensors):
    domain = np.zeros((domain_width,domain_width))
    for x,y in np.random.randint(0,domain_width,(nb_sensors,2)):
        domain[y][x] = 1
    return domain

def make_init(init,**kwargs):
    def f():
        return init(**kwargs)
    return f


########################################################################
# Neighborhood
########################################################################

def neighb_num_rect(sol, scale):
    return np.random.random(len(sol)) * scale - scale/2

def neighb_bit_rect(sol, scale):
    # Copy, because Python pass by reference.
    new = copy.copy(sol)
    for yy in range(len(sol)):
        for xx in range(len(sol[yy])):
            if sol[yy][xx] == 1:
                new[yy][xx] = 0
                d = np.random.randint(-scale//2,scale//2,2)
                new[yy+y(d)][xx+x(d)] = 1
    return new

def make_neig(neighb,**kwargs):
    def f(sol):
        return neighb(sol, **kwargs)
    return f


########################################################################
# Stopping criterions
########################################################################

def iters_nb(val,sol,nb_it):
    for i in range(nb_it):
        yield i
    yield i

def make_iter(iters,nb_it):
    def cont(val,sol):
        return iters(val,sol,nb_it)
    return cont


########################################################################
# Algorithms
########################################################################

def search(func, init, neighb, iters):
    best_sol = init()
    best_val = func(best_sol)
    for i in iters(best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        if val > best_val:
            best_val = val
            best_sol = sol
    return val,sol

if __name__=="__main__":

    d = 2
    nb_sensors = 3
    sensor_range = 2
    domain_width = 10

    # domain = np.zeros((domain_width,domain_width))
    # domain = coverage(domain,[(10,50),(40,80)],50)
    # plt.imshow(domain)
    # plt.show()

    print(
            search(
                make_func(cover_num, domain_width=domain_width, sensor_range=sensor_range),
                make_init(rand_num, dim=d * nb_sensors),
                make_neig(neighb_num_rect, scale=domain_width/10),
                make_iter(iters_nb,10)
            )
        )

    print(
            search(
                make_func(cover_bit, domain_width=domain_width, sensor_range=sensor_range),
                make_init(rand_bit, domain_width=domain_width, nb_sensors=nb_sensors),
                make_neig(neighb_bit_rect, scale=3),
                make_iter(iters_nb,10)
            )
        )

