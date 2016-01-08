pyDive
======

Distributed Interactive Visualization and Exploration of large datasets.

[![Build Status master](https://img.shields.io/travis/ComputationalRadiationPhysics/pyDive/master.svg?label=master)](https://travis-ci.org/ComputationalRadiationPhysics/pyDive/branches)
[![Build Status dev](https://img.shields.io/travis/ComputationalRadiationPhysics/pyDive/dev.svg?label=dev)](https://travis-ci.org/ComputationalRadiationPhysics/pyDive/branches)
[![pypi version](https://img.shields.io/pypi/v/pyDive.svg)](https://pypi.python.org/pypi/pyDive)
[![Number of pyDive Downloads](https://img.shields.io/pypi/dm/pyDive.svg)](https://pypi.python.org/pypi/pyDive/)
[![pyDive license](https://img.shields.io/pypi/l/pyDive.svg)](#software-license)

## What is pyDive?

Use pyDive to work with homogeneous, n-dimensional arrays that are too big to fit into your local machine's memory.
pyDive provides containers whose elements are distributed across a cluster or stored in
a large hdf5/adios-file if the cluster is still too small. All computation and data-access is then done in parallel by the cluster nodes in the background. 
If you feel like working with [numpy](http://www.numpy.org) arrays pyDive has reached the goal!

pyDive is developed and maintained by the **[Junior Group Computational Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132&pOid=30354)**
at the [Institute for Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132)
at [HZDR](http://www.hzdr.de/).

**Features:**
 - Since all cluster management is given to [IPython.parallel](http://ipython.org/ipython-doc/dev/parallel/) you can take your
   existing profiles for pyDive. No further cluster configuration needed.
 - Save bandwidth by slicing an array in parallel on disk first before loading it into main memory!
 - GPU-cluster array available thanks to [pycuda](http://mathema.tician.de/software/pycuda/) with additional support for non-contiguous memory.
 - As all of pyDive's distributed array types are auto-generated from local arrays like numpy, hdf5, pycuda, etc... 
   you can easily make your own local array classes distributed too.

## Dive in!

```python
import pyDive
pyDive.init(profile='mpi')

h5field = pyDive.h5.open("myData.h5", "myDataset", distaxes=(0,1))
ones = pyDive.ones_like(h5field)

# Distribute file i/o and computation across the cluster
h5field[::10,:] = h5field[::10,:].load() + 5.0 * ones[::10,:]
```

## Documentation

In our [Online Documentation](http://ComputationalRadiationPhysics.github.io/pyDive/), [pdf](http://ComputationalRadiationPhysics.github.io/pyDive/pyDive.pdf) you can find 
detailed information on all interfaces as well as some [Tutorials](http://computationalradiationphysics.github.io/pyDive/tutorial.html)
and a [Quickstart](http://computationalradiationphysics.github.io/pyDive/start.html).

## Software License

pyDive is licensed under the **LGPLv3+**.
Licences can be found in [COPYING](COPYING) and [COPYING.LESSER](COPYING.LESSER), respectively.
