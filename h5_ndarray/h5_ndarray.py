from mpi4py import MPI
import h5py as h5
from ndarray import ndarray, ndarray_factories
from ndarray import ndarray_helper as helper
import IPParallelClient as com

h5_ndarray_id = 0

class h5_ndarray(object):
    def __init__(self, h5_filename, dataset_path, distaxis, max_elements_node=0):
        self.h5_filename = h5_filename
        self.dataset_path = dataset_path

        self.dataset = h5.File(h5_filename, 'r')[dataset_path]

        self.shape = list(self.dataset.shape)
        self.distaxis = distaxis
        self.max_elements_node = max_elements_node
        self.dtype = self.dataset.dtype

        # generate a unique variable name used on the target representing this instance
        global h5_ndarray_id
        self.name = 'h5_ndarray' + str(h5_ndarray_id)
        h5_ndarray_id += 1

        # create dataset object on each target
        self.dataset_name = self.name + "_dataset"
        self.fileHandle_name = self.name + "_file"
        view = com.getView()
        view.execute("%s = h5.File('%s', 'r', driver='mpio', comm=MPI.COMM_WORLD)"\
            % (self.fileHandle_name, h5_filename))
        view.execute("%s = %s['%s']"\
            % (self.dataset_name, self.fileHandle_name, dataset_path))

        # ndarray representing the cached sub-part of the hdf5 dataset residing in RAM
        #self.chunk = None
        # offset of chunk in respect to the hdf5 dataset in the direction of chunkaxis
        #self.offset = 0

    def __del__(self):
        view = com.getView()
        view.execute("%s.close()" % self.fileHandle_name)

    def __getitem__(self, args):
        if args ==  slice(None):
            args = (slice(None) for i in range(len(self.shape)))

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [args]

        assert len(self.shape) == len(args)

        # single data access is not allowed
        assert not all(type(arg) is int for arg in args),\
            "single data access is not allowed"

        new_shape, clean_slices = helper.subWindow_of_shape(self.shape, args)

        print "new_shape: ", new_shape
        print "clean_slices: ", clean_slices

        # result ndarray
        result = ndarray_factories.hollow(new_shape, self.distaxis, dtype=self.dtype)

        print "result.idx_ranges: ", result.idx_ranges

        # create local slice objects for each engine
        local_slices = helper.createLocalSlices(clean_slices, self.distaxis, result.idx_ranges)

        print "local_slices: ", local_slices

        # scatter slice objects to the engines
        view = com.getView()
        view.scatter('window', local_slices, targets=result.targets_in_use)

        view.execute('%s = %s[tuple(window[0])]' % (result.name, self.dataset_name), targets=result.targets_in_use)
        return result

    def __setitem__(self, key, value):
        assert isinstance(value, ndarray.ndarray), "assigned array has to be a pyDive ndarray"

        if key == slice(None):
            key = [sliceNone] * len(self.shape)

        # singe data access is not allowed
        assert not all(type(key_comp) is int for key_comp in key),\
            "single data access is not allowed"

        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        assert len(key) == len(self.shape)

        new_shape, clean_slices = helper.subWindow_of_shape(self.shape, key)

        assert new_shape == value.shape

        # create local slice objects for each engine
        local_slices = helper.createLocalSlices(clean_slices, value.distaxis, value.idx_ranges)

        # scatter slice objects to the engines
        view = com.getView()
        view.scatter('window', local_slices, targets=value.targets_in_use)

        # write 'value' to disk in parallel
        view.execute('%s[tuple(window[0])] = %s' % (self.dataset_name, repr(value)), \
            targets=value.targets_in_use)
