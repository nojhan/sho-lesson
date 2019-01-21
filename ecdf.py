import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher


def guess_number_evals(filenames):
    """Guess the number of evals from first file."""
    with open(filenames[0], 'r') as fd:
        nevals = len(fd.readlines())
    return nevals


def along_runtime(filenames, data):
    for fid,filename in enumerate(filenames):
        with open(filename, 'r') as fd:
            strdata = csv.reader(fd, delimiter=';')
            for i,row in enumerate(strdata):
                evals = int(row[0])
                val = float(row[1])
                data[evals,fid] = val
    return data


def cumul(data, delta, optim = None, do_min = False):
    # Keep only best values along columns.
    for i in range(1,len(data)):
        for j in range(len(data[i])):
            data[i,j] = max( data[i,j], data[i-1,j] )

    if not optim:
        optim = data.max()

    # Normalize.
    norm = data/optim

    # Threshold.
    if do_min:
        ecdf = (norm < delta)
    else:
        ecdf = (norm > delta)

    # Sum across rows.
    return ecdf.sum(axis=1)/data.shape[1]


def parse(filenames, delta, nb_rows = None, optim = None, do_min = False):
    if not nb_rows:
        nb_rows = guess_number_evals(filenames)

    data = np.zeros( (nb_rows+1, len(filenames)) )
    data = along_runtime(filenames,data)
    ert = cumul(data, delta, optim, do_min)
    return ert


def make_name(names, delta, erts, name_strip = [], do_min = False):
    common = names[0]
    for run in names:
        match = SequenceMatcher(None, common, run).find_longest_match(0, len(common), 0, len(run))
        common = common[match.a: match.a + match.size]

    for strp in name_strip:
        common = common.replace(strp,"")

    name = u"{} $\Delta={}$".format(common,delta)

    if name in erts:
        i += 1
        name += " ({})".format(i)

    return name


if __name__ == "__main__":


    can = argparse.ArgumentParser()

    can.add_argument("-e", "--evals", metavar="NB", default=None, type=int,
            help="Max number of evaluations to consider")

    # can.add_argument("-q", "--quality", action='store_true',
    #         help="Produce Expected Quality ECDF, instead of Expected Runtime ECDF.")

    can.add_argument("-m", "--min", action='store_true',
            help="Minimization problem, instead of maximization.")

    can.add_argument("-o", "--optimum", metavar="VAL", default=None, type=float,
            help="Best value used for normalization (else, default to the max in the data).")

    can.add_argument("-s", "--name-strip", metavar="STR", default=[],
            type=str, action='append',
            help="Remove this string from the labels.")

    can.add_argument("-d", "--delta", metavar="PERC",
            action='append', type=float, required=True,
            help="Target(s), as a percentage of values normalized against optimum.")

    can.add_argument("-r", "--runs", metavar="FILES", nargs='*', required=True, action='append')

    the = can.parse_args()

    erts = {}
    names = []
    i = 0
    data_max = -1*float("inf")

    if not the.evals:
        nb_rows = guess_number_evals(the.runs[0])
    data = np.zeros( (nb_rows+1, len(the.runs[0])) )

    if the.optimum:
        data_max = the.optimum
    else:
        sys.stderr.write("Compute max:\n")
        i = 0
        for runs in the.runs:
            for delta in the.delta:
                i += 1
                sys.stderr.write( "\r{}/{}".format(i,len(the.runs)*len(the.delta)) )
                data = along_runtime(runs,data)
                data_max = max(data_max, data.max())

    sys.stderr.write("\nCompute ECDFs:\n")
    i = 0
    for runs in the.runs:
        for delta in the.delta:
            i += 1
            sys.stderr.write( "\r{}/{}".format(i,len(the.runs)*len(the.delta)) )
            ert = parse(
                    runs, delta,
                    nb_rows = the.evals, optim = data_max, do_min = the.min
                )

            name = make_name(runs, delta, erts, the.name_strip, the.min)
            erts[name] = ert
    sys.stderr.write("\nPlot\n")

    fig = plt.figure()
    for name in erts:
        plt.plot(erts[name], label=name)

    plt.ylim([0,1])

    if the.min:
        comp = "<"
    else:
        comp=">"
    # plt.ylabel(r"$P\left(f\left(\hat{x})\right)/"+str(the.optimum)+comp+r"\Delta\right)$")
    plt.ylabel(r"$P\left(f\left(\hat{x}\right)/"+str(data_max)+comp+r"\Delta\right)$")
    plt.xlabel("Time (#function evals)")
    plt.title("Expected RunTime Empirical Cumulative Density Function ({} runs)".format(str(len(the.runs[0]))))
    plt.legend()
    plt.show()

