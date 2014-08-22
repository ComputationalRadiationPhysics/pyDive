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

#from mpi4py import MPI
import h5py as h5
from ..ndarray import ndarray, factories
from ..ndarray import helper
from .. import IPParallelClient as com
import numpy as np

h5_ndarray_id = 0

class h5_ndarray(object):
    """Represents a single hdf5-dataset like a virtual, cluster-wide array.
        Data access goes through array slicing where data is written to respectively read from
        :class:`pyDive.ndarray.ndarray.ndarray` objects in parallel using all engines.

        Example: ::

            h5_data = pyDive.h5.fromPath(<file_path>, "/data/0/fields/FieldB/x", distaxis=0)
            data = h5_data[:] # read the entire dataset into engine-memory
            data = data**2
            h5_data[:] = data # write everything back

    """
    def __init__(self, h5_filename, dataset_path, distaxis=0, window=None):
        """Creates an :class:`h5_ndarray` instance.
        By using this method you may only load a single dataset. If you want to
        load a *structure* of datasets at once see :func:`pyDive.h5_ndarray.factories.fromPath`.

        :param str h5_filename: Path of the hdf5-file
        :param str dataset_path: Path to the dataset within the hdf5-file
        :param int distaxis: axis on which dataset is to be distributed over during data-access
        :param window: This param let you specify a sub-part of the array as a virtual container.
            Example: window=np.s_[:,:,::2]
        :type window: list of slice objects (:obj:`numpy.s_`).

        Notes:
            - The dataset's attributes are stored in ``h5array.attrs``.
        """
        self.h5_filename = h5_filename
        self.dataset_path = dataset_path
        self.dataset = h5.File(h5_filename)[dataset_path]
        self.attrs = self.dataset.attrs
        #: axis of element-distribution
        self.distaxis = distaxis
        #: datatype of a single data value
        self.dtype = self.dataset.dtype

        self.arraytype = self.__class__

        if not window:
            window = [slice(None)] * len(self.dataset.shape)
        self.shape, self.window = helper.subWindow_of_shape(self.dataset.shape, window)

        #: total bytes consumed by the elements of the array.
        self.nbytes = self.dtype.itemsize * np.prod(self.shape)

        # generate a unique variable name used on target representing this instance
        global h5_ndarray_id
        self.name = 'h5_ndarray' + str(h5_ndarray_id)
        h5_ndarray_id += 1

        # create dataset object on each target
        self.dataset_name = self.name + "_dataset"
        self.fileHandle_name = self.name + "_file"
        view = com.getView()
        view.execute("%s = h5.File('%s', 'r')"\
            % (self.fileHandle_name, h5_filename))
        view.execute("%s = %s['%s']"\
            % (self.dataset_name, self.fileHandle_name, dataset_path))

    def __del__(self):
        view = com.getView()
        view.execute("%s.close()" % self.fileHandle_name)

    def __getitem__(self, args):
        if args == slice(None):
            args = (slice(None),) * len(self.shape)

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [args]

        assert len(self.shape) == len(args)

        assert not all(type(arg) is int for arg in args),\
            "single data access is not allowed"

        result_shape, clean_slices = helper.subWindow_of_shape(self.shape, args)

        # Applying 'clean_slices' after 'self.window' results in 'total_slices'
        total_slices = [slice(self.window[i].start + clean_slices[i].start * self.window[i].step,\
                              self.window[i].start + clean_slices[i].stop * self.window[i].step,\
                              self.window[i].step * clean_slices[i].step) for i in range(len(args))]

        # result ndarray
        result = factories.hollow(result_shape, self.distaxis, dtype=self.dtype)

        # create local slice objects for each engine
        local_slices = helper.createLocalSlices(total_slices, self.distaxis, result.idx_ranges)

        # scatter slice objects to the engines
        view = com.getView()
        view.scatter('window', local_slices, targets=result.targets_in_use)
        view.execute('%s = %s[tuple(window[0])]' % (repr(result), self.dataset_name), targets=result.targets_in_use)

        return result

    def __setitem__(self, key, value):
        assert isinstance(value, ndarray.ndarray), "assigned array has to be a pyDive ndarray"

        if key == slice(None):
            key = [sliceNone] * len(self.shape)

        assert not all(type(key_comp) is int for key_comp in key),\
            "single data access is not allowed"

        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        assert len(key) == len(self.shape)

        new_shape, clean_slices = helper.subWindow_of_shape(self.shape, key)

        # Applying 'clean_slices' after 'self.window'-slices result in 'total_slices'
        total_slices = [slice(self.window[i].start + clean_slices[i].start * self.window[i].step,\
                              self.window[i].start + clean_slices[i].stop * self.window[i].step,\
                              self.window[i].step * clean_slices[i].step) for i in range(len(key))]

        assert new_shape == value.shape

        # create local slice objects for each engine
        local_slices = helper.createLocalSlices(clean_slices, value.distaxis, value.idx_ranges)

        # scatter slice objects to the engines
        view = com.getView()
        view.scatter('window', local_slices, targets=value.targets_in_use)

        # write 'value' to disk in parallel
        view.execute('%s[tuple(window[0])] = %s' % (self.dataset_name, repr(value)), \
            targets=value.targets_in_use)

    def __str__(self):
        return "<hdf5 dset: " + self.dataset_path + ", " + str(self.shape) + ", " + str(self.dtype) + ">"
