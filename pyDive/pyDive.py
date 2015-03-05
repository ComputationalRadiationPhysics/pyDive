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

__doc__=\
"""Make most used functions and modules directly accessable from pyDive."""

# ndarray
import arrays.ndarray
globals().update(arrays.ndarray.factories)
globals().update(arrays.ndarray.ufuncs)
from arrays.ndarray import ndarray

# hdf5
try:
    import arrays.h5_ndarray as h5
except ImportError:
    pass

# adios
try:
    import arrays.ad_ndarray as adios
except ImportError:
    pass

# cloned_ndarray
import cloned_ndarray.factories
cloned = cloned_ndarray.factories

# fragmentation
from fragment import fragment

## algorithm
import algorithm
map = algorithm.map
reduce = algorithm.reduce
mapReduce = algorithm.mapReduce

# particle-mesh mappings
import mappings
mesh2particles = mappings.mesh2particles
particles2mesh = mappings.particles2mesh

# arrayOfStructs
import arrayOfStructs
arrayOfStructs = arrayOfStructs.arrayOfStructs

# picongpu
import picongpu
picongpu = picongpu

# init
import IPParallelClient
init = IPParallelClient.init


# module doc
items = [item for item in globals().items() if not item[0].startswith("__")]
items.sort(key=lambda item: item[0])

fun_names = [":obj:`" + item[0] + "<" + item[1].__module__ + "." + item[0] + ">`"\
    for item in items if hasattr(item[1], "__module__")]

import inspect
module_names = [":mod:`" + item[0] + "<" + item[1].__name__ + ">`"\
    for item in items if inspect.ismodule(item[1])]

__doc__ += "\n\n**Functions**:\n\n" + "\n\n".join(fun_names)\
    + "\n\n**Modules**:\n\n" + "\n\n".join(module_names)
