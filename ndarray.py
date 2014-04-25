import numpy as np
import ndarray_helper as helper
import IPParallelClient as com
import dist_math

ndarray_id = 0

class ndarray(object):
    def __init__(self, shape, distaxis, dtype=np.float, idx_ranges=None, targets_in_use=None, no_allocation=False):
        self.shape = list(shape)
        self.dtype = dtype
        self.distaxis = distaxis

        assert distaxis >= 0 and distaxis < len(self.shape)

        if idx_ranges is None and targets_in_use is None:
            # number of available targets (engines)
            num_targets_av = len(view.targets)

            # shape of the mpi-local ndarray
            localshape = np.array(self.shape)
            localshape[distaxis] = (self.shape[distaxis] - 1) / num_targets_av + 1
            tmp = localshape[distaxis]

            # number of occupied targets by this ndarray instance
            num_targets = (self.shape[distaxis] - 1) / localshape[distaxis] + 1

            # list of pairs on which each pair stores the range of indices [begin, end) for the distributed axis on each target
            # this is the decomposition of the distributed axis
            self.idx_ranges = [(r * tmp, (r+1) * tmp) for r in range(0, num_targets-1)]
            self.idx_ranges += [((num_targets-1) * tmp, self.shape[distaxis])]
            # list of indices of the occupied targets
            self.targets_in_use = list(range(num_targets))
        elif idx_ranges is not None and targets_in_use is not None:
            self.idx_ranges = idx_ranges[:]
            self.targets_in_use = targets_in_use[:]
        else:
            raise ValueError("either args 'idx_ranges' and 'targets_in_use' have to be given both or not given both.")

        # generate a unique variable name used on the target representing this instance
        global ndarray_id
        self.name = 'dist_ndarray' + str(ndarray_id)
        ndarray_id += 1

        if no_allocation:
            view.push({self.name : None}, targets=self.targets_in_use)
        else:
            # instanciate an empty ndarray object of the appropriate shape on each target in use
            localshapes = [self.shape[:] for i in range(len(self.targets_in_use))]
            for i in range(len(self.targets_in_use)):
                localshapes[i][distaxis] = self.idx_ranges[i][1] - self.idx_ranges[i][0]

            view.scatter('localshape', localshapes, targets=self.targets_in_use)
            view.push({'dtype' : dtype}, targets=self.targets_in_use)
            view.execute('%s = empty(localshape[0], dtype=dtype)' % self.name, targets=self.targets_in_use)

    def __del__(self):
        view.execute('del %s' % self.name, targets=self.targets_in_use)

    def __getitem__(self, args):
        # if args is [:] then return a hard copy of the entire ndarray
        if args == slice(None):
            return self.copy()

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = (args,)

        assert len(args) == len(self.shape)

        # if args is a list of indices return a single data value
        if all(type(arg) is int for arg in args):
            dist_idx = args[self.distaxis]
            for i in range(len(self.idx_ranges)):
                begin, end = self.idx_ranges[i]
                if dist_idx >= end: continue
                local_idx = list(args)
                local_idx[self.distaxis] = dist_idx - begin
                return view.pull("%s%s" % (self.name, repr(local_idx)), targets=self.targets_in_use[i])

        # shape of the new sliced ndarray
        new_shape = helper.sliceShape(args, self.shape)

        # create a clean, wrapped slice object for the distributed axis
        if type(args[self.distaxis]) is int:
            distaxis_slice = slice(args[self.distaxis], args[self.distaxis]+1)
        else:
            distaxis_slice = args[self.distaxis]
        wrapped_ids = distaxis_slice.indices(self.shape[self.distaxis])
        distaxis_slice = slice(*wrapped_ids)

        # determine properties of the new sliced ndarray
        new_idx_ranges = []
        new_targets_in_use = []
        local_slices = []
        total_ids = 0
        for i in range(len(self.idx_ranges)):
            begin, end = self.idx_ranges[i]
            # first index within [begin, end) after slicing
            firstSubIdx = helper.getFirstSubIdx(distaxis_slice, begin, end)
            if firstSubIdx is None: continue
            # calculate last index of distaxis_slice
            tmp = (distaxis_slice.stop-1 - distaxis_slice.start) / distaxis_slice.step
            lastIdx = distaxis_slice.start + tmp * distaxis_slice.step
            # calculate last sub index
            tmp = (end-1 - firstSubIdx) / distaxis_slice.step
            lastSubIdx = firstSubIdx + tmp * distaxis_slice.step
            lastSubIdx = min(lastSubIdx, lastIdx)
            # slice object on the current target
            local_slices.append(slice(firstSubIdx - begin, lastSubIdx+1 - begin, distaxis_slice.step))
            # number of indices remaining on the current target after slicing
            num_ids = (lastSubIdx - firstSubIdx) / distaxis_slice.step + 1
            # index range for the current target
            new_idx_ranges.append([total_ids, total_ids + num_ids])
            total_ids += num_ids
            # target id
            new_targets_in_use.append(self.targets_in_use[i])

        result = ndarray(new_shape, self.distaxis, self.dtype, new_idx_ranges, new_targets_in_use, True)

        # remote slicing
        args = list(args)
        for i in range(len(local_slices)):
            args[self.distaxis] = local_slices[i]
            view.push({'args' : args}, targets=new_targets_in_use[i])
        view.execute('%s = %s[args]' % (result.name, self.name), targets=new_targets_in_use)

        return result

    def __setitem__(self, key, value):
        # if args is [:] then assign value to the entire ndarray
        if key == slice(None):
            if isinstance(value, np.ndarray):
                value = array(value, self.distaxis)
            other = value.dist_like(self)
            view.execute("%s[:] = %s" % (self.name, other.name), targets=self.targets_in_use)
            return

        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        assert len(key) == len(self.shape)

        # value assignment (key == list of indices)
        if all(type(k) is int for k in key):
            dist_idx = key[self.distaxis]
            for i in range(len(self.idx_ranges)):
                begin, end = self.idx_ranges[i]
                if dist_idx >= end: continue
                local_idx = list(key)
                local_idx[self.distaxis] = dist_idx - begin
                view.push({'value' : value}, targets=self.targets_in_use[i])
                view.execute("%s%s = value" % (self.name, repr(local_idx)), targets=self.targets_in_use[i])
                return

        # assign value to sub-array of self
        sub_array = self[key]
        sub_array[:] = value

    def __str__(self):
        return self.gather().__str__()

    def gather(self):
        local_arrays = view.pull(self.name, targets=self.targets_in_use)
        return np.concatenate(local_arrays, axis=self.distaxis)

    def copy(self):
        result = ndarray(self.shape, self.distaxis, self.dtype, self.idx_ranges, self.targets_in_use, True)
        view.execute("%s = %s[:]" % (result.name, self.name), targets=self.targets_in_use)
        return result

    def dist_like(self, other):
        assert self.shape == other.shape
        assert self.distaxis == other.distaxis # todo: add this feature

        # if self is already distributed like other do nothing
        if self.targets_in_use == other.targets_in_use:
            if self.idx_ranges == other.idx_ranges:
                return self

        # communication meta-data
        src_targets = [[]]
        src_tags = [[]]
        src_distaxis_sizes = [[]]
        dest_targets = [[]]
        dest_tags = [[]]
        dest_distaxis_sizes = [[]]

        self_idx = 0
        other_idx = 0
        tag = 0
        begin = 0
        # loop the common decomposition of self and other.
        # the current partition is given by [begin, end)
        while not begin == self.shape[self.distaxis]:
            end = min(self.idx_ranges[self_idx][1], other.idx_ranges[other_idx][1])

            # source's meta-data
            src_targets[self_idx].append(other.targets_in_use[other_idx])
            src_tags[self_idx].append(tag)
            src_distaxis_sizes[self_idx].append(end - begin)

            # destination's meta-data
            dest_targets[other_idx].append(self.targets_in_use[self_idx])
            dest_tags[other_idx].append(tag)
            dest_distaxis_sizes[other_idx].append(end - begin)

            # go to the next common partition
            if end == self.idx_ranges[self_idx][1]:
                if not end == self.shape[self.distaxis]:
                    src_targets.append([])
                    src_tags.append([])
                    src_distaxis_sizes.append([])
                self_idx += 1
            if end == other.idx_ranges[other_idx][1]:
                if not end == self.shape[self.distaxis]:
                    dest_targets.append([])
                    dest_tags.append([])
                    dest_distaxis_sizes.append([])
                other_idx += 1

            begin = end
            tag += 1

        # push communication meta-data to the targets
        view.scatter('src_targets', src_targets, targets=self.targets_in_use)
        view.scatter('src_tags', src_tags, targets=self.targets_in_use)
        view.scatter('src_distaxis_sizes', src_distaxis_sizes, targets=self.targets_in_use)

        view.scatter('dest_targets', dest_targets, targets=other.targets_in_use)
        view.scatter('dest_tags', dest_tags, targets=other.targets_in_use)
        view.scatter('dest_distaxis_sizes', dest_distaxis_sizes, targets=other.targets_in_use)

        # result ndarray
        result = ndarray(self.shape, self.distaxis, self.dtype, other.idx_ranges, other.targets_in_use)

        # send
        view.execute('scatterArrayMPI_async(%s, src_targets[0], src_tags[0], src_distaxis_sizes[0], %d, target2rank)' \
            % (self.name, self.distaxis), targets=self.targets_in_use)

        # receive
        view.execute('gatherArraysMPI_sync(%s, dest_targets[0], dest_tags[0], dest_distaxis_sizes[0], %d, target2rank)' \
            % (result.name, self.distaxis), targets=result.targets_in_use)

        return result

    # numerical operators
    def __add__(self, other):
        return dist_math.binary_op(self, other, '+')
    def __sub__(self, other):
        return dist_math.binary_op(self, other, '-')
    def __mul__(self, other):
        return dist_math.binary_op(self, other, '*')
    def __div__(self, other):
        return dist_math.binary_op(self, other, '/')
    def __floordiv__(self, other):
        return dist_math.binary_op(self, other, '//')
    def __pow__(self, other):
        return dist_math.binary_op(self, other, '**')
    def __pos__(self):
        return dist_math.unary_op(self, '+')
    def __neg__(self):
        return dist_math.unary_op(self, '-')
    # numerical in-place operators
    def __iadd__(self, other):
        return dist_math.binary_iop(self, other, '+=')
    def __isub__(self, other):
        return dist_math.binary_iop(self, other, '-=')
    def __imul__(self, other):
        return dist_math.binary_iop(self, other, '*=')
    def __idiv__(self, other):
        return dist_math.binary_iop(self, other, '/=')
    def __ifloordiv__(self, other):
        return dist_math.binary_iop(self, other, '//=')
    def __ipow__(self, other):
        return dist_math.binary_iop(self, other, '**=')

def array(array_like, distaxis):
    # numpy array
    if isinstance(array_like, np.ndarray):
        # result ndarray
        result = ndarray(array_like.shape, distaxis, array_like.dtype, no_allocation=True)

        tmp = np.rollaxis(array_like, distaxis)
        sub_arrays = [tmp[begin:end] for begin, end in result.idx_ranges]
        # roll axis back
        sub_arrays = [np.rollaxis(ar, 0, distaxis+1) for ar in sub_arrays]

        view.scatter('sub_array', sub_arrays, targets=result.targets_in_use)
        view.execute("%s = sub_array[0].copy()" % result.name, targets=result.targets_in_use)

        return result

def empty_like(a):
    return ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use)

dist_math.empty_like = empty_like
dist_math.ndarray = ndarray

com.init()
view = com.getView()
a = ndarray((4,3), distaxis=0)

a[:] = np.ones(a.shape) * -1.0

a[1:3, 1:3] = np.arange(2*2).reshape((2,2)).astype(a.dtype)

a[-1,0] = 5.0

a += a**2

print a