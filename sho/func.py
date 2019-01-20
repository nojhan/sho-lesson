
########################################################################
# Wrappers around objective functions
########################################################################

class Dump:
    """A wrapper around an objective function that
    dumps a line in a file every time the objective function is called."""

    def __init__(self, func, filename="run.csv", fmt="{it} ; {val} ; {sol}\n", sepsol=" , "):
        self.func = func
        self.filename = filename
        self.fmt = fmt
        self.sepsol = sepsol
        self.counter = 0
        # Erase previous file.
        with open(self.filename, 'w') as fd:
            fd.write("")

    def __call__(self, sol):
        val = self.func(sol)
        self.counter += 1
        with open(self.filename, 'a') as fd:
            fmtsol = self.sepsol.join([str(i) for i in sol])
            fd.write( self.fmt.format(it=self.counter, val=val, sol=fmtsol) )
        return val
