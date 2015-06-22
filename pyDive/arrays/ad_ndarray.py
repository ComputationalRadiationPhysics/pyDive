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
import adios as ad
from pyDive.arrays.ndarray import hollow_like
import pyDive.distribution.multiple_axes as multiple_axes
import pyDive.IPParallelClient as com
from itertools import islice
from .. import structured
import pyDive.arrays.local.ad_ndarray

ad_ndarray = multiple_axes.distribute(pyDive.arrays.local.ad_ndarray.ad_ndarray, "ad_ndarray", "ad_ndarray", may_allocate=False)

def load(self):
    """Load array from file into main memory of all engines in parallel.

    :return: pyDive.ndarray instance
    """
    result = hollow_like(self)
    view = com.getView()
    view.execute("{0} = {1}.load()".format(result.name, self.name), targets=result.target_ranks)
    return result
ad_ndarray.load = load
del load

def open_variable(filename, variable_path, distaxis=0):
    """Create a pyDive.adios.ad_ndarray instance from file.

    :param filename: name of adios file.
    :param variable_path: path within adios file to a single variable.
    :param distaxis int: distributed axis
    :return: pyDive.adios.ad_ndarray instance
    """
    fileHandle = ad.file(filename)
    variable = fileHandle.var[variable_path]
    dtype = variable.type
    shape = tuple(variable.dims)
    fileHandle.close()

    result = ad_ndarray(shape, dtype, distaxis, None, None, True)

    target_shapes = result.target_shapes()
    target_offset_vectors = result.target_offset_vectors()

    view = com.getView()
    view.scatter("shape", target_shapes, targets=result.target_ranks)
    view.scatter("offset", target_offset_vectors, targets=result.target_ranks)
    view.execute("{0} = pyDive.arrays.local.ad_ndarray.ad_ndarray('{1}','{2}',shape=shape[0],offset=offset[0])"\
        .format(result.name, filename, variable_path), targets=result.target_ranks)

    return result

def open(filename, datapath, distaxis=0):
    """Create a pyDive.adios.ad_ndarray instance respectively a structure of
    pyDive.adios.ad_ndarray instances from file.

    :param filename: name of adios file.
    :param datapath: path within adios file to a single variable or a group of variables.
    :param distaxis int: distributed axis
    :return: pyDive.adios.ad_ndarray instance
    """
    fileHandle = ad.file(filename)
    variable_paths = fileHandle.var.keys()
    fileHandle.close()

    def update_tree(tree, variable_path, variable_path_iter, leaf):
        node = variable_path_iter.next()
        if node == leaf:
            tree[leaf] = open_variable(filename, variable_path, distaxis)
            return
        tree[node] = {}
        update_tree(tree[node], variable_path, variable_path_iter, leaf)

    n = len(datapath.split("/"))

    structOfArrays = {}
    for variable_path in variable_paths:
        if not variable_path.startswith(datapath):
            continue
        path_nodes = variable_path.split("/")
        path_nodes_it = iter(path_nodes)

        # advance 'path_nodes_it' n times
        next(islice(path_nodes_it, n, n), None)

        update_tree(structOfArrays, variable_path, path_nodes_it, path_nodes[-1])

    return structured.structured(structOfArrays)
