"""
Copyright 2015 Heiko Burau

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

import numpy as np
import pyDive.distribution.multiple_axes as multiple_axes
from pyDive.distribution.interengine import GPU_copier
import pyDive.arrays.local.gpu_ndarray

gpu_ndarray = multiple_axes.distribute(pyDive.arrays.local.gpu_ndarray.gpu_ndarray, "gpu_ndarray",\
    "pyDive.arrays.local.gpu_ndarray", interengine_copier=GPU_copier)

factories = multiple_axes.generate_factories(gpu_ndarray, ("empty", "zeros"), np.float)
factories.update(multiple_axes.generate_factories_like(gpu_ndarray, ("empty_like", "zeros_like")))
globals().update(factories)

def ones(shape, dtype=np.float, distaxes='all', **kwargs):
    result = zeros(shape, dtype, distaxes, **kwargs)
    result += 1
    return result

def ones_like(other, **kwargs):
    result = zeros_like(other, **kwargs)
    result += 1
    return result

import pyDive.IPParallelClient as com
import pyDive.arrays.ndarray

def to_cpu(self):
    """Copy array data to cpu main memory.

    :result pyDive.ndarray: distributed cpu array.
    """
    result = pyDive.arrays.ndarray.hollow_like(self)
    view = com.getView()
    view.execute("{0} = {1}.to_cpu()".format(result.name, self.name), targets=result.target_ranks)
    return result
gpu_ndarray.to_cpu = to_cpu
del to_cpu

def hollow(shape, dtype=np.float, distaxes='all'):
    """Create a pyDive.gpu_ndarray instance distributed across all engines without allocating a local
    gpu-array.

    :param ints shape: shape of array
    :param dtype: datatype of a single element
    :param ints distaxes: distributed axes. Defaults to 'all' meaning each axis is distributed.
    """
    return gpu_ndarray(shape, dtype, distaxes, None, None, True)

def hollow_like(other):
    """Create a pyDive.gpu_ndarray instance with the same
    shape, distribution and type as ``other`` without allocating a local gpu-array.
    """
    return gpu_ndarray(other.shape, other.dtype, other.distaxes, other.target_offsets, other.target_ranks, True)

def array(array_like, distaxes='all'):
    """Create a pyDive.gpu_ndarray instance from an array-like object.

    :param array_like: Any object exposing the array interface, e.g. numpy-array, python sequence, ...
    :param ints distaxis: distributed axes. Defaults to 'all' meaning each axis is distributed.
    """
    result_cpu = pyDive.arrays.ndarray.array(array_like, distaxes)
    result = hollow_like(result_cpu)
    view = com.getView()
    view.execute("{0} = pyDive.arrays.local.gpu_ndarray.gpu_ndarray_cast(pycuda.gpuarray.to_gpu({1}))"\
        .format(repr(result), repr(result_cpu)), targets=result.target_ranks)

    return result

#ufunc_names = [key for key, value in np.__dict__.items() if isinstance(value, np.ufunc)]
#ufuncs = multiple_axes.generate_ufuncs(ufunc_names, "np")

#globals().update(ufuncs)