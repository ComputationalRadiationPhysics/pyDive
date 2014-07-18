from mpi4py import MPI
import h5py as h5
from ndarray import ndarray, factories
from ndarray import helper
import IPParallelClient as com
import numpy as np

import debug

h5_ndarray_id = 0

class h5_ndarray(object):
    def __init__(self, h5_filename, dataset_path, distaxis, window=None):
        self.h5_filename = h5_filename
        self.dataset_path = dataset_path
        self.dataset = h5.File(h5_filename)[dataset_path]
        self.distaxis = distaxis
        self.dtype = self.dataset.dtype

        if not window:
            window = [slice(None)] * len(self.dataset.shape)
        self.shape, self.window = helper.subWindow_of_shape(self.dataset.shape, window)

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
        if args ==  slice(None):
            args = (slice(None),) * len(self.shape)

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [args]

        assert len(self.shape) == len(args)

        assert not all(type(arg) is int for arg in args),\
            "single data access is not allowed"

        result_shape, clean_slices = helper.subWindow_of_shape(self.shape, args)

        # Applying 'clean_slices' after 'self.window'-slices result in 'total_slices'
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
