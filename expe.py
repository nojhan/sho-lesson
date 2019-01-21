import os
import random
import itertools
import subprocess

class Run:
    def __init__(self, cmd_tpl):
        self.cmd_tpl = cmd_tpl

    def single(self, doe):
        for params in doe:
            cmd = self.cmd_tpl.format(**params)
            subprocess.run(cmd, shell=True)


    def all(self, doe):
        queue = []
        for params in doe:
            cmd = self.cmd_tpl.format(**params)
            p = subprocess.Popen(cmd, shell=True)
            queue.append(p)

        for p in queue:
            p.wait()


    def batch(self, doe, batch_size = None):
        if batch_size is None:
            batch_size = os.cpu_count()

        nb = 0
        queue = []
        for params in doe:
            if nb >= batch_size:
                for p in queue:
                    p.wait()
                queue.clear()
                nb = 0
            else:
                cmd = self.cmd_tpl.format(**params)
                p = subprocess.Popen(cmd, shell=True)
                queue.append(p)
                nb += 1
        # in case batch_size = float("inf")
        for p in queue:
            p.wait()


    def qsub(self, doe):
        for params in doe:
            cmd = self.cmd_tpl.format(**params)
            print("TODO qsub",cmd)


class Expe:
    def __init__(self, **kwargs):
        self.names = []
        self.axes = []
        self.static_params = kwargs

    def __iter__(self):
        for p in itertools.product( *(self.axes) ):
            params = self.static_params
            for i in range(len(self.names)):
                params[self.names[i]] = p[i]
            yield params

    def forall(self, name, iters):
        self.names.append(name)
        self.axes.append(iters)
        return self

    def random(self, name, nb, rmin=0, rmax=1):
        """Note: this intentionally produce the SAME random sequence for each product of axes."""
        self.names.append(name)
        def rand():
            for i in range(nb):
                yield random.random() * (rmax-rmin) + rmin
        self.axes.append(rand())
        return self



if __name__ == "__main__":

    const_args=" --nb-sensors 5 --sensor-range 0.2 --domain-width 50 --iters 10000"
    solvers = ["num_greedy","bit_greedy","num_rand","bit_rand"]
    nbruns = 2
    outdir = "results"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cmd = "echo \"--no-plot --dir {outdir} --seed {seed} --solver {solver}\""
    # for seed in range(nbruns):
    #     for params in expe({"seed":[seed]}):
    #         for by_four in batchsplit(solvers, 4):
    #             batchrun("solver", by_four, params, cmd_tpl)


    doe = Expe(outdir=outdir).forall("seed",range(nbruns)).forall("solver",solvers)

    Run(cmd).single(doe)
    Run(cmd).all(doe)
    Run(cmd).batch(doe)
    Run(cmd).qsub(doe)
