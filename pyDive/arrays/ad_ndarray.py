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

import adios as ad
from pyDive.arrays.ndarray import hollow_like
import pyDive.distribution.generic_array as generic_array
import pyDive.ipyParallelClient as com
from pyDive.structured import structured
import pyDive.arrays.local.ad_ndarray

ad_ndarray = generic_array.distribute(pyDive.arrays.local.ad_ndarray.ad_ndarray, "ad_ndarray", "ad_ndarray", may_allocate=False)

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

def open_variable(filename, variable_path, distaxes='all'):
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
    view.execute("{0} = pyDive.arrays.local.ad_ndarray.ad_ndarray('{1}','{2}',shape=shape[0],offset=offset[0])"\
        .format(repr(result), filename, variable_path), targets=result.ranks())

    return result

def open(filename, datapath, distaxes='all'):
    """Create a pyDive.adios.ad_ndarray instance respectively a structure of
    pyDive.adios.ad_ndarray instances from file.

    :param filename: name of adios file.
    :param datapath: path within adios file to a single variable or a group of variables.
    :param distaxes ints: distributed axes. Defaults to 'all' meaning each axis is distributed.
    :return: pyDive.adios.ad_ndarray instance
    """
    fileHandle = ad.file(filename)
    variable_paths = list(fileHandle.var.keys())
    fileHandle.close()

    pairs = []

    datapath_nodes = datapath.strip("/").split("/")
    for var_path in variable_paths:
        var_path_nodes = var_path.strip("/").split("/")
        # if lists matches exactly return a single *ad_ndarray* instance
        if datapath_nodes == var_path_nodes:
            return open_variable(filename, var_path, distaxes)

        # if *var_path* includes *datapath* add it to the tree
        if datapath_nodes == var_path_nodes[:len(datapath_nodes)]:
            path = "/".join(var_path_nodes[len(datapath_nodes):])
            array = open_variable(filename, var_path, distaxes)
            pairs.append((path, array))

    assert pairs, "{} does not have variable: {}".format(filename, datapath)

    return structured(pairs)