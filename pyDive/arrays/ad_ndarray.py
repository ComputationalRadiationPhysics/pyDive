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

try:
    import adios as ad
except ImportError:
    pass
import pyDive.distribution.helper as helper
import numpy as np

class ad_ndarray_local(object):

    def __init__(self, filename, variable_path, shape=None, window=None, offset=None):
        self.filename = filename
        self.variable_path = variable_path

        fileHandle = ad.file(filename)
        variable = fileHandle.var[variable_path]
        self.dtype = variable.type
        if shape is None:
            shape = variable.dims
        self.shape = tuple(shape)
        fileHandle.close()

        if window is None:
            window = [slice(0, s, 1) for s in shape]
        self.window = tuple(window)
        if offset is None:
            offset = (0,) * len(shape)
        self.offset = tuple(offset)

        #: total bytes consumed by the elements of the array.
        self.nbytes = np.dtype(self.dtype).itemsize * np.prod(self.shape)

    def load(self):
        begin = []
        size = []
        for o, w in zip(self.offset, self.window):
            if type(w) is int:
                begin.append(o+w)
                size.append(1)
            else:
                begin.append(o+w.start)
                size.append(w.stop-w.start)

        fileHandle = ad.file(self.filename)
        variable = fileHandle.var[self.variable_path]
        result = variable.read(tuple(begin), tuple(size))
        fileHandle.close()

        # remove all single dimensional axes unless they are a result of slicing, i.e. a[n,n+1]
        single_dim_axes = [axis for axis in range(len(self.window)) if type(self.window[axis]) is int]
        if single_dim_axes:
            result = np.squeeze(result, axis=single_dim_axes)

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
            "single data access is not supported by adios"

        assert all(arg.step == 1 or arg.step == None for arg in args if type(arg) is slice),\
            "strided access in not supported by adios"

        result_shape, clean_view = helper.view_of_shape(self.shape, args)

        # Applying 'clean_view' after 'self.window', results in 'result_window'
        result_window = helper.view_of_view(self.window, clean_view)

        return ad_ndarray_local(self.filename, self.variable_path, result_shape, result_window, self.offset)


import os
onTarget = os.environ.get("onTarget", 'False')

# execute this code only if it is not executed on engine
if onTarget == 'False':
    from pyDive.arrays.ndarray import hollow_like
    import pyDive.distribution.single_axis as single_axis
    import pyDive.IPParallelClient as com
    from itertools import islice
    from .. import arrayOfStructs

    ad_ndarray = single_axis.distribute(ad_ndarray_local, "ad_ndarray", "ad_ndarray", may_allocate=False)

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
        view.execute("{0} = ad_ndarray.ad_ndarray_local('{1}','{2}',shape=shape[0],offset=offset[0])"\
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

        return arrayOfStructs.arrayOfStructs(structOfArrays)
