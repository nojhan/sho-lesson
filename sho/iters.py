import sys

########################################################################
# Stopping criterions
########################################################################

def max(i, val, sol, nb_it):
    if i < nb_it:
        return True
    else:
        return False

# Stopping criterions that are actually just checkpoints.

def several(i, val, sol, agains):
    """several  several stopping criterions in one."""
    over = []
    for again in agains:
        over.append( again(i, val, sol) )
    return all(over)


def save(i, val, sol, filename="run.csv", fmt="{it} ; {val} ; {sol}\n"):
    """Save all iterations to a file."""
    # Append a line at the end of the file.
    with open(filename.format(it=i), 'a') as fd:
        fd.write( fmt.format(it=i, val=val, sol=sol) )
    return True # No incidence on termination.


def history(i, val, sol, history):
    history.append((val,sol))
    return True


def log(i, val, sol, fmt="{it} {val}\n"):
    """Print progress on stderr."""
    sys.stderr.write( fmt.format(it=i, val=val) )
    return True

