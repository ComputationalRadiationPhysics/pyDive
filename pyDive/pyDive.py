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
#import mappings
#mesh2particles = mappings.mesh2particles
#particles2mesh = mappings.particles2mesh

# arrayOfStructs
import arrayOfStructs
arrayOfStructs = arrayOfStructs.arrayOfStructs

# picongpu
import picongpu
picongpu = picongpu

# init
import IPParallelClient as com
init = com.init