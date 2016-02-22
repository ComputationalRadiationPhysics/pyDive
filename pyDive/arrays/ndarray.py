# Copyright 2014-2016 Heiko Burau
#
# This file is part of pyDive.
#
# pyDive is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pyDive.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pyDive.distribution.generic_array as generic_array
from pyDive.distribution.interengine import MPI_copier

ndarray = generic_array.distribute(
    local_arraytype=np.ndarray,
    newclassname="ndarray",
    target_modulename="np",
    interengine_copier=MPI_copier)

factories = generic_array.generate_factories(ndarray, ("empty", "zeros", "ones"), np.float)
factories.update(generic_array.generate_factories_like(ndarray, ("empty_like", "zeros_like", "ones_like")))

empty = factories["empty"]
zeros = factories["zeros"]
ones = factories["ones"]
empty_like = factories["empty_like"]
zeros_like = factories["zeros_like"]
ones_like = factories["ones_like"]


def array(array_like, distaxes='all'):
    """Create a pyDive.ndarray instance from an array-like object.

    :param array_like: Any object exposing the array interface, e.g. numpy-array, python sequence, ...
    :param ints distaxes: distributed axes. Defaults to 'all' meaning each axis is distributed.
    """
    np_array = np.array(array_like)
    result = empty(np_array.shape, np_array.dtype, distaxes)
    result[:] = np_array
    return result


def hollow(shape, dtype=np.float, distaxes='all'):
    """Create a pyDive.ndarray instance distributed across all engines without allocating a local
    numpy-array.

    :param ints shape: shape of array
    :param dtype: datatype of a single element
    :param ints distaxes: distributed axes. Defaults to 'all' meaning each axis is distributed.
    """
    return ndarray(shape, dtype, distaxes, None, None, True)


def hollow_like(other):
    """Create a pyDive.ndarray instance of the same
    shape, distribution and dtype as ``other`` without allocating a local numpy-array.
    """
    return ndarray(other.shape, other.dtype, other.distaxes, other.decomposition, True)

factories.update({"array": array, "hollow": hollow, "hollow_like": hollow_like})

ufunc_names = [key for key, value in np.__dict__.items() if isinstance(value, np.ufunc)]
ufuncs = generic_array.generate_ufuncs(ufunc_names, "np")

globals().update(ufuncs)
