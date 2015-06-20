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
import h5py as h5
from pyDive.arrays.ndarray import hollow_like
import pyDive.distribution.multiple_axes as multiple_axes
import pyDive.IPParallelClient as com
from .. import structured
import pyDive.arrays.local.h5_ndarray

h5_ndarray = multiple_axes.distribute(pyDive.arrays.local.h5_ndarray.h5_ndarray, "h5_ndarray", "h5_ndarray", may_allocate=False)

def load(self):
    """Load array from file into main memory of all engines in parallel.

    :return: pyDive.ndarray instance
    """
    result = hollow_like(self)
    view = com.getView()
    view.execute("{0} = {1}.load()".format(result.name, self.name), targets=result.target_ranks)
    return result
h5_ndarray.load = load
del load

def open_dset(filename, dataset_path, distaxes='all'):
    """Create a pyDive.h5.h5_ndarray instance from file.

    :param filename: name of hdf5 file.
    :param dataset_path: path within hdf5 file to a single dataset.
    :param distaxes ints: distributed axes. Defaults to 'all' meaning each axis is distributed.
    :return: pyDive.h5.h5_ndarray instance
    """
    fileHandle = h5.File(filename, "r")
    dataset = fileHandle[dataset_path]
    dtype = dataset.dtype
    shape = dataset.shape
    fileHandle.close()

    result = h5_ndarray(shape, dtype, distaxes, None, None, True)

    target_shapes = result.target_shapes()
    target_offset_vectors = result.target_offset_vectors()

    view = com.getView()
    view.scatter("shape", target_shapes, targets=result.target_ranks)
    view.scatter("offset", target_offset_vectors, targets=result.target_ranks)
    view.execute("{0} = pyDive.arrays.local.h5_ndarray.h5_ndarray('{1}','{2}',shape=shape[0],offset=offset[0])"\
        .format(result.name, filename, dataset_path), targets=result.target_ranks)

    return result

def open(filename, datapath, distaxes='all'):
    """Create an pyDive.h5.h5_ndarray instance respectively a structure of
    pyDive.h5.h5_ndarray instances from file.

    :param filename: name of hdf5 file.
    :param dataset_path: path within hdf5 file to a single dataset or hdf5 group.
    :param distaxes ints: distributed axes. Defaults to 'all' meaning each axis is distributed.
    :return: pyDive.h5.h5_ndarray instance / structure of pyDive.h5.h5_ndarray instances
    """
    hFile = h5.File(filename, 'r')
    datapath = datapath.rstrip("/")
    group_or_dataset = hFile[datapath]
    if type(group_or_dataset) is not h5._hl.group.Group:
        # dataset
        return open_dset(filename, datapath, distaxes)

    def create_tree(group, tree, dataset_path):
        for key, value in group.items():
            # group
            if type(value) is h5._hl.group.Group:
                tree[key] = {}
                create_tree(value, tree[key], dataset_path + "/" + key)
            # dataset
            else:
                tree[key] = open_dset(filename, dataset_path + "/" + key, distaxes)

    group = group_or_dataset
    structOfArrays = {}
    create_tree(group, structOfArrays, datapath)
    return structured.structured(structOfArrays)
