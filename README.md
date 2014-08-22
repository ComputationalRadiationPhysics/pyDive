pyDive
======

Distributed Interactive Visualization and Exploration of large datasets.

## What is pyDive?

Use pyDive to work with homogeneous, n-dimensional arrays that are too big to fit into your local machine's memory.
pyDive provides array-interfaces representing virtual containers whose elements are distributed across an MPI-cluster or stored in
a large hdf5-file if the cluster is still too small. All computation and data-access is then done in parallel by the cluster nodes in the background. 
If you feel like working with [numpy](http://www.numpy.org) arrays pyDive has reached the goal!

pyDive is developed and maintained by the **[Junior Group Computational Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132&pOid=30354)**
at the [Institute for Radiation Physics](http://www.hzdr.de/db/Cms?pNid=132)
at [HZDR](http://www.hzdr.de/).

Designed to ease the analysis of simulation data coming from the [picongpu project](https://github.com/ComputationalRadiationPhysics/picongpu) 
pyDive also holds functions for particle-mesh mapping and supports structured datatypes.

## Dive in!

```python
import pyDive
pyDive.init(profile='mpi')

h5field = pyDive.h5.fromPath("myData.h5", "myDataset", distaxis=0)
ones = pyDive.ones_like(h5field)

# Distribute file i/o and computation across the cluster
h5field[1:] = h5field[1:] - h5field[:-1] + 5.0 * ones[1:]
```

## Documentation

In our [Online Documentation](http://ComputationalRadiationPhysics.github.io/pyDive/), [pdf](http://ComputationalRadiationPhysics.github.io/pyDive/pyDive.pdf) you can find 
detailed information on all interfaces as well as some [Tutorials](http://computationalradiationphysics.github.io/pyDive/tutorial.html)
and a [Quickstart](http://computationalradiationphysics.github.io/pyDive/start.html).

## Software License

pyDive is licensed under the **GPLv3+ and LGPLv3+** (it is *dual licensed*).
Licences can be found in [GPL](COPYING) or [LGPL](COPYING.LESSER), respectively.
