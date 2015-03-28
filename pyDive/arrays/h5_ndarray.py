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

import h5py as h5
import pyDive.distribution.helper as helper
import numpy as np

class h5_ndarray_local(object):

    def __init__(self, filename, dataset_path, shape=None, window=None, offset=None):
        self.filename = filename
        self.dataset_path = dataset_path

        fileHandle = h5.File(filename, "r")
        dataset = fileHandle[dataset_path]
        #self.attrs = dataset.attrs
        #: datatype of a single data value
        self.dtype = dataset.dtype
        if shape is None:
            shape = dataset.shape
        self.shape = tuple(shape)
        fileHandle.close()

        if window is None:
            window = [slice(0, s, 1) for s in shape]
        self.window = tuple(window)
        if offset is None:
            offset = (0,) * len(shape)
        self.offset = tuple(offset)

        #: total bytes consumed by the elements of the array.
        self.nbytes = self.dtype.itemsize * np.prod(self.shape)

    def load(self):
        window = list(self.window)
        for i in range(len(window)):
            if type(window[i]) is int:
                window[i] += self.offset[i]
            else:
                window[i] = slice(window[i].start + self.offset[i], window[i].stop + self.offset[i], window[i].step)

        fileHandle = h5.File(self.filename, "r")
        dataset = fileHandle[self.dataset_path]
        result = dataset[tuple(window)]
        fileHandle.close()
        return result

    def __getitem__(self, args):
        if args == slice(None):
            args = (slice(None),) * len(self.shape)

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [args]

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(args), len(self.shape))

        assert not all(type(arg) is int for arg in args),\
            "single data access is not supported"

        result_shape, clean_view = helper.view_of_shape(self.shape, args)

        # Applying 'clean_view' after 'self.window', results in 'result_window'
        result_window = helper.view_of_view(self.window, clean_view)

        return h5_ndarray_local(self.filename, self.dataset_path, result_shape, result_window, self.offset)

import os
onTarget = os.environ.get("onTarget", 'False')

# execute this code only if it is not executed on engine
if onTarget == 'False':
    from pyDive.arrays.ndarray import hollow_like
    import pyDive.distribution.single_axis as single_axis
    import pyDive.IPParallelClient as com
    from .. import arrayOfStructs

    h5_ndarray = single_axis.distribute(h5_ndarray_local, "h5_ndarray", "h5_ndarray", may_allocate=False)

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

    def open_dset(filename, dataset_path, distaxis=0):
        """Create a pyDive.h5.h5_ndarray instance from file.

        :param filename: name of hdf5 file.
        :param dataset_path: path within hdf5 file to a single dataset.
        :param distaxis int: distributed axis
        :return: pyDive.h5.h5_ndarray instance
        """
        fileHandle = h5.File(filename, "r")
        dataset = fileHandle[dataset_path]
        dtype = dataset.dtype
        shape = dataset.shape
        fileHandle.close()

        result = h5_ndarray(shape, dtype, distaxis, None, None, True)

        target_shapes = result.target_shapes()
        target_offset_vectors = result.target_offset_vectors()

        view = com.getView()
        view.scatter("shape", target_shapes, targets=result.target_ranks)
        view.scatter("offset", target_offset_vectors, targets=result.target_ranks)
        view.execute("{0} = h5_ndarray.h5_ndarray_local('{1}','{2}',shape=shape[0],offset=offset[0])"\
            .format(result.name, filename, dataset_path), targets=result.target_ranks)

        return result

    def open(filename, datapath, distaxis=0):
        """Create an pyDive.h5.h5_ndarray instance respectively a structure of
        pyDive.h5.h5_ndarray instances from file.

        :param filename: name of hdf5 file.
        :param dataset_path: path within hdf5 file to a single dataset or hdf5 group.
        :param distaxis int: distributed axis
        :return: pyDive.h5.h5_ndarray instance / structure of pyDive.h5.h5_ndarray instances
        """
        hFile = h5.File(filename, 'r')
        datapath = datapath.rstrip("/")
        group_or_dataset = hFile[datapath]
        if type(group_or_dataset) is not h5._hl.group.Group:
            # dataset
            return open_dset(filename, datapath, distaxis)

        def create_tree(group, tree, dataset_path):
            for key, value in group.items():
                # group
                if type(value) is h5._hl.group.Group:
                    tree[key] = {}
                    create_tree(value, tree[key], dataset_path + "/" + key)
                # dataset
                else:
                    tree[key] = open_dset(filename, dataset_path + "/" + key, distaxis)

        group = group_or_dataset
        structOfArrays = {}
        create_tree(group, structOfArrays, datapath)
        return arrayOfStructs.arrayOfStructs(structOfArrays)
