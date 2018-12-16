import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

from sho import *

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
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = the.iters),
                    make.iter(iters.save,
                        filename = the.solver+".csv",
                        fmt = "{it} ; {val} ; {sol}\n"),
                    make.iter(iters.log,
                        fmt="\r{it} {val}")
                ]
            )

    # Erase the previous file.
    with open(the.solver+".csv", 'w') as fd:
        fd.write("# {} {}\n".format(the.solver,the.domain_width))

    val,sol,sensors = None,None,None
    if the.solver == "num_greedy":
        val,sol = algo.greedy(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range * the.domain_width),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.domain_width/10), # TODO think of an alternative.
                iters
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_greedy":
        val,sol = algo.greedy(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.domain_width/10),
                iters
            )
        sensors = bit.to_sensors(sol)


    # Fancy output.
    print("\n",val,":",sensors)

    domain = pb.coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    plt.imshow(domain)
    plt.show()
