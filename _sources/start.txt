Getting started
===============

Quickstart
----------

pyDive is built on top of *IPython.parallel*, *numpy*, *mpi4py* and *h5py*. Running ``python setup.py install`` will install
pyDive with these and other required packages from `requirements.txt`. Alternatively you can install it via pip: ``pip install pyDive``.

Basic code example: ::

  import pyDive
  pyDive.init()

  arrayA = pyDive.ones([1000, 1000, 1000], distaxis=0)
  arrayB = pyDive.zeros_like(arrayA)

  # do some array operations, + - * / sin cos, ..., slicing, etc...
  ...

  # get numpy-array
  result = arrayC.gather()
  # plot result
  ...

Before actually running this script there must have been an IPython.parallel cluster launched (see section below) otherwise `pyDive.init()` fails.

To keep things simple pyDive distributes array-memory only along **one** user-specified axis. This axis is given by the `distaxis`
parameter at array instanciation. It should usually be the largest axis in order to have the best surface-to-volume ratio. 
But keep in mind that during arithmetic operations both arrays have to be distributed along the *same* axis.

Although the array elements are stored on the cluster nodes you have full access through indexing. If you want to have a numpy-array
from a pyDive-array anyway you can call the method ``arrayC.gather()`` but make sure that your pyDive-array is small enough to fit
into your local machine's memory. If not you may want to slice it first.

.. _cluster-config:

Setup an IPython.parallel cluster configuration
-----------------------------------------------

The first step is to create an IPython.parallel profile: http://ipython.org/ipython-doc/2/parallel/parallel_process.html.
The name of this profile is the argument of :func:`pyDive.init`. It defaults to ``"mpi"``.
Starting the cluster is then the second and final step::

  $ ipcluster start -n 4 --profile=mpi

Run tests
---------

In order to test the pyDive installation you can run::

  $ python setup.py test

This will ask you for the IPython.parallel profile to be used and the number of engines to be started, e.g.: ::

  $ Name of your IPython-parallel profile you want to run the tests with: pbs
  $ Number of engines: 256

Then the script starts the cluster, runs the tests and finally stops the cluster. If you have already a cluster running by your own
you can also run the tests by launching ``py.test`` from the pyDive directory and setting the environment variable ``IPP_PROFILE_NAME``
to the profile's name.

Overview
--------

pyDive knows three kinds of arrays associated to a separate python package respectively:
  - :obj:`pyDive.ndarray` -> Stores array elements in cluster nodes' memory.
  - :obj:`pyDive.h5_ndarray` -> Stores array elements in a hdf5-file.
  - :obj:`pyDive.cloned_ndarray` -> Holds independent copies of one array on cluster nodes.

Among these three packages there are a few modules:
  - :mod:`pyDive.arrayOfStructs` -> structured datatypes
  - :mod:`pyDive.algorithm` -> map, reduce, mapReduce
  - :mod:`pyDive.mappings` -> particle-mesh mappings
  - :mod:`pyDive.picongpu` -> helper functions for picongpu-users
  - :mod:`pyDive.pyDive` -> shortcuts for most used functions

