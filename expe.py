import subprocess
import itertools


def batchsplit(axe, njobs):
    for i in range(0,len(axe),njobs):
        yield axe[i:i+njobs]


def batchrun(key, jobs, params, cmd_tpl):
        procs = []
        for job in jobs:
            params[key] = job
            cmd = cmd_tpl.format(**params)
            p = subprocess.Popen(cmd, shell=True)
            procs.append(p)

        for job in procs:
            job.wait()


def expe(axes):
    for p in itertools.product(*[axes[k] for k in axes]):
        params = {}
        for i in range(len(axes)):
            params[list(axes.keys())[i]] = p[i]
        yield params



if __name__ == "__main__":
    import os
    import subprocess

    const_args=" --nb-sensors 5 --sensor-range 0.2 --domain-width 50 --iters 10000"
    solvers = ["num_greedy","bit_greedy","num_rand","bit_rand"]
    nbruns = 2
    outdir = "results"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cmd_tpl = "python3 snp.py --no-plot --dir {outdir} --seed {{seed}} --solver {{solver}}".format(outdir=outdir)
    for seed in range(nbruns):
        for params in expe({"seed":[seed]}):
            for by_four in batchsplit(solvers, 4):
                batchrun("solver", by_four, params, cmd_tpl)


