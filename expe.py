
if __name__ == "__main__":
    import os
    import subprocess

    # can = argparse.ArgumentParser()
    #
    # can.add_argument("-n", "--nb-sensors", metavar="NB", default=3, type=int,
    #         help="Number of sensors")
    #
    # can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
    #         help="Sensors' range (as a fraction of domain width)")
    #
    # can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
    #         help="Domain width (a number of cells)")
    #
    # can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
    #         help="Maximum number of iterations")
    #
    # the = can.parse_args()

    const_args=" --nb-sensors 5 --sensor-range 0.2 --domain-width 50 --iters 10000"
    solvers = ["num_greedy","bit_greedy","num_rand","bit_rand"]
    nbruns = 100
    outdir = "results"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for seed in range(nbruns):
        procs = []
        for solver in solvers:
            print(seed,solver)
            p = subprocess.Popen(
                    "python3 snp.py "
                    + const_args
                    + " --no-plot --dir {} --seed {} --solver {}"
                    .format(outdir,seed,solver),
                    shell=True
                )
            procs.append(p)

        for proc in procs:
            proc.wait()


