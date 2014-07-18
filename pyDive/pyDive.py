# make most used functions directly accessable from here
# ------------------------------------------------------

# ndarray
from ndarray import ndarray
from ndarray import factories
hollow = ndarray.factories.hollow
empty = ndarray.factories.empty
hollow_like = ndarray.factories.hollow_like
empty_like = ndarray.factories.empty_like
array = ndarray.factories.array

# h5_ndarray
from h5_ndarray import factories
h5 = h5_ndarray.factories

# cloned_ndarray
from cloned_ndarray import factories
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