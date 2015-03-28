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

import numpy as np
import pyDive.distribution.single_axis as single_axis
from pyDive.distribution.interengine import MPI_copier

ndarray = single_axis.distribute(np.ndarray, "ndarray", "np", interengine_copier=MPI_copier)

factories = single_axis.generate_factories(ndarray, ("empty", "zeros", "ones"), np.float)
factories.update(single_axis.generate_factories_like(ndarray, ("empty_like", "zeros_like", "ones_like")))

globals().update(factories)

def array(array_like, distaxis=0):
    """Create a pyDive.ndarray instance from an array-like object.

    :param array_like: Any object exposing the array interface, e.g. numpy-array, python sequence, ...
    :param int distaxis: distributed axis
    """
    np_array = np.array(array_like)
    result = empty(np_array.shape, np_array.dtype, distaxis)
    result[:] = np_array
    return result

def hollow(shape, dtype=np.float, distaxis=0):
    """Create a pyDive.ndarray instance distributed across all engines without allocating a local
    numpy-array.

    :param ints shape: shape of array
    :param dtype: datatype of a single element
    :param int distaxis: distributed axis
    """
    return ndarray(shape, dtype, distaxis, None, None, True)

def hollow_like(other):
    """Create a pyDive.ndarray instance with the same
    shape, distribution and type as ``other`` without allocating a local numpy-array.
    """
    return ndarray(other.shape, other.dtype, other.distaxis, other.target_offsets, other.target_ranks, True)

factories.update({"array" : array, "hollow" : hollow, "hollow_like" : hollow_like})

ufunc_names = [key for key, value in np.__dict__.items() if isinstance(value, np.ufunc)]
ufuncs = single_axis.generate_ufuncs(ufunc_names, "np")

globals().update(ufuncs)