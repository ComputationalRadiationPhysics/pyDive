from IPython.parallel import Client
from IPython.parallel import interactive

view = None

def init():
    #init direct view
    global view

    view = Client(profile='mpi')[:]
    view.block = True
    view.execute('from numpy import *')
    view.execute('from mpi4py import MPI')
    view.execute('import os')
    view.run('ndarray/interengine.py')

    get_rank = interactive(lambda: MPI.COMM_WORLD.Get_rank())
    all_ranks = view.apply(get_rank)
    view['target2rank'] = all_ranks

def getView():
    global view
    return view