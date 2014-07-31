pyDive
======

Distributed Interactive Visualization and Exploration of large datasets.

## Why pyDive?

Use pyDive to work with homogeneous, n-dimensional arrays that are too big to fit into your local machine's memory.
pyDive provides array-interfaces representing virtual containers whose elements may be distributed across an MPI-cluster or stored in
a large hdf5-file. If you feel like working with numpy arrays pyDive has reached the goal!

Designed to ease the analysis of simulation data coming from the picongpu project pyDive also holds functions for particle-mesh mapping
and supports structured datatypes.

## Dive in!

```python
import pyDive
pyDive.init(profile='mpi')
```
