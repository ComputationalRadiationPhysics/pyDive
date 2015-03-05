"""
Copyright 2014 Heiko Burau

This file is part of pyDive.

pyDive is free software: you can redistribute it and/or modify
it under the terms of of either the GNU General Public License or
the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
pyDive is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License and the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU General Public License
and the GNU Lesser General Public License along with pyDive.
If not, see <http://www.gnu.org/licenses/>.
"""
__doc__ = None

from IPython.parallel import Client
from IPython.parallel import interactive
from collections import Counter
import sys

#: IPParallel direct view
view = None
#: number of processes per node
ppn = None

def init(profile='mpi'):
    """Initialize pyDive.

    :param str profile: The name of the cluster profile of *IPython.parallel*. Has to be an MPI-profile.\
        Defaults to 'mpi'.
    """
    # init direct view
    global view

    client = Client(profile=profile)
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
        os.environ["onTarget"] = 'True'
        from pyDive import arrayOfStructs
        from pyDive import algorithm
        from pyDive.distribution import interengine
        try:
            import pyDive.arrays.h5_ndarray as h5_ndarray
        except ImportError:
            pass
        try:
            import pyDive.arrays.ad_ndarray as ad_ndarray
        except ImportError:
            pass
         ''')

    # get number of processes per node (ppn)
    def hostname():
        import socket
        return socket.gethostname()
    hostnames = view.apply(interactive(hostname))
    global ppn
    ppn = max(Counter(hostnames).values())

    # mpi ranks
    get_rank = interactive(lambda: MPI.COMM_WORLD.Get_rank())
    all_ranks = view.apply(get_rank)
    view['target2rank'] = all_ranks

def getView():
    global view
    assert view is not None, "pyDive.init() has not been called yet."
    return view

def getPPN():
    global ppn
    assert ppn is not None, "pyDive.init() has not been called yet."
    return ppn
