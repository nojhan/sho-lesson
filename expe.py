import os
import stat
import random
import itertools
import subprocess


class HowLocal:
    def __init__(self, doe, job, directory):
        self.doe = doe
        self.job = job
        self.dir = directory # TODO

    def one_by_one(self):
        for params in self.doe:
            cmd = self.job(params)
            subprocess.run(cmd, shell=True)

    def all(self):
        queue = []
        for params in self.doe:
            cmd = self.job(params)
            p = subprocess.Popen(cmd, shell=True)
            queue.append(p)

        for p in queue:
            p.wait()


    def batch(self, batch_size = None):
        if batch_size is None:
            batch_size = os.cpu_count()

        nb = 0
        queue = []
        for params in self.doe:
            if nb >= batch_size:
                for p in queue:
                    p.wait()
                queue.clear()
                nb = 0
            else:
                cmd = self.job(params)
                p = subprocess.Popen(cmd, shell=True)
                queue.append(p)
                nb += 1
        # in case batch_size = float("inf")
        for p in queue:
            p.wait()


class HowGridEngine:
    def __init__(self, doe, job, directory):
        self.doe = doe
        self.job = job
        self.dir = directory


    def qsub(self, *flags, **kwargs):
        for params in self.doe:

            # Handy name
            name = "_".join([str(params[k]) for k in params])

            # Add default qsub name if not asked.
            if "N" not in kwargs:
                kwargs["N"] = name

            # Default output in results directory.
            if "o" not in kwargs:
                kwargs["o"] = self.dir
            if "e" not in kwargs:
                kwargs["e"] = self.dir

            qsub = ["qsub"]
            # Add qsub flags
            for arg in flags:
                qsub.append("-{}".format(arg))
            # Add qsub options
            for k in kwargs:
                qsub.append("-{opt} {val}".format(opt=k, val = kwargs[k]))

            # Get the user's command.
            cmd = self.job(params)

            # Put it into a script.
            script=os.path.join(self.dir,"run_{}.sh".format(name))
            with open(script,'w') as fd:
                fd.write("#!/bin/sh\n") # Shebang
                fd.write(""">&2 echo "{}"\n""".format(cmd) ) # Log
                fd.write("{}\n".format(cmd)) # Actual command

            # Make script executable.
            st = os.stat(script)
            os.chmod(script, st.st_mode | stat.S_IEXEC) # chmod u+x

            # Call qsub.
            qsub.append(script)
            # print(" ".join(qsub))
            subprocess.run(" ".join(qsub), shell=True)


class Where:
    def __init__(self, doe, job):
        self.doe = doe
        self.job = job

    def __make_dir(self,directory):
        if not directory:
            directory = datetime.datetime.now().isoformat().replace(":","-")

        try:
            os.mkdir(directory)
        except FileExistsError:
            pass


    def on_computer(self, directory=None):
        self.__make_dir(directory)
        return HowLocal(self.doe, self.job, directory)

    def on_grid_engine(self, directory=None):
        self.__make_dir(directory)
        return HowGridEngine(self.doe, self.job, directory)


class run:
    def __init__(self, doe):
        self.doe = doe

    def command(self, cmd):
        def job(params):
            return cmd.format(**params)
        return Where(self.doe, job)

    def func(self, job):
        return Where(self.doe, job)

    # def script(self, script):
    #     pass


class plan:
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
    outdir = "tests"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    cmd = "echo \"--no-plot --dir {outdir} --seed {seed} --solver {solver}\""


    doe = plan(outdir=outdir).forall("seed",range(nbruns)).forall("solver",solvers)

    # run(doe).command(cmd).on_computer().one_by_one()
    # run(doe).command(cmd).on_computer().all()
    # run(doe).command(cmd).on_computer().batch(4)

    # def job(params):
    #     return cmd.format(**params)
    # run(doe).func(job).on_grid_engine(directory="tests").qsub("cwd",S="/bin/bash")

    run(doe).command(cmd).on_grid_engine(outdir).qsub("cwd",S="/bin/bash")
