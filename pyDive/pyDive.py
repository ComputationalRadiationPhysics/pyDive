"""Make most used functions and modules directly accessable from pyDive."""

# ndarray
import ndarray.ndarray
import ndarray.factories
hollow = ndarray.factories.hollow
empty = ndarray.factories.empty
zeros = ndarray.factories.zeros
ones = ndarray.factories.ones
hollow_like = ndarray.factories.hollow_like
empty_like = ndarray.factories.empty_like
zeros_like = ndarray.factories.zeros_like
ones_like = ndarray.factories.ones_like
array = ndarray.factories.array

# h5_ndarray
import h5_ndarray.factories
h5 = h5_ndarray.factories

# cloned_ndarray
import cloned_ndarray.factories
cloned = cloned_ndarray.factories

# algorithm
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
