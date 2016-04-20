# Copyright 2015-2016 Heiko Burau
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

import adios as ad
from pyDive.arrays.ndarray import hollow_like
import pyDive.distribution.generic_array as generic_array
import pyDive.ipyParallelClient as com
import pyDive.arrays.local.ad_ndarray

ad_ndarray = generic_array.distribute(
    local_arraytype=pyDive.arrays.local.ad_ndarray.ad_ndarray,
    newclassname="ad_ndarray",
    target_modulename="ad_ndarray",
    may_allocate=False)


def load(self):
    """Load array from file into main memory of all engines in parallel.

    :return: pyDive.ndarray instance
    """
    result = hollow_like(self)
    view = com.getView()
    view.execute("{0} = {1}.load()".format(result.name, self.name), targets=result.decomposition.ranks)
    return result
ad_ndarray.load = load
del load


def open(filename, variable_path, distaxes='all'):
    """Create a pyDive.adios.ad_ndarray instance from file.

    :param filename: name of adios file.
    :param variable_path: path within adios file to a single variable.
    :param distaxes ints: distributed axes. Defaults to 'all' meaning each axis is distributed.
    :return: pyDive.adios.ad_ndarray instance
    """
    fileHandle = ad.file(filename)
    variable = fileHandle.var[variable_path]
    dtype = variable.dtype
    shape = tuple(map(int, variable.dims))
    fileHandle.close()

    result = ad_ndarray(shape, dtype, distaxes, None, True)

    target_shapes = result.target_shapes()
    target_offset_vectors = result.target_offset_vectors()

    view = com.getView()
    view.scatter("shape", target_shapes, targets=result.ranks())
    view.scatter("offset", target_offset_vectors, targets=result.ranks())
    view.execute(
        """{0} = pyDive.arrays.local.ad_ndarray.ad_ndarray(
            '{1}','{2}',
            shape=shape[0],
            offset=offset[0])""".format(repr(result), filename, variable_path),
        targets=result.ranks())

    return result
