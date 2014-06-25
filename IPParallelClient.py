from IPython.parallel import Client
from IPython.parallel import interactive
import sys

view = None

def init():
    #init direct view
    global view

    client = Client(profile='mpi')
    client.clear()
    view = client[:]
    view.block = True
    view.execute('''\
        import numpy as np
        from mpi4py import MPI
        import h5py as h5
        import os, sys
        import psutil
        import math
        import onTarget
        onTarget.onTarget = True''')

    get_rank = interactive(lambda: MPI.COMM_WORLD.Get_rank())
    all_ranks = view.apply(get_rank)
    view['target2rank'] = all_ranks

def getView():
    global view
    return view

init()