# -*- coding: utf-8 -*-
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
import pyDive.IPParallelClient as com
import helper
from collections import defaultdict

array_id = 0

class DistributedGenericArray(object):
    """
    Represents a cluster-wide, multidimensional, homogeneous array of fixed-size elements.
    *cluster-wide* means that its elements are distributed across IPython.parallel-engines.
    The distribution is done in one or multiply dimensions along user-specified axes.
    The user can optionally specify which engine maps to which index range or leave the default
    that persuits an uniform distribution across all engines.

    This **{arraytype_name}** - class is auto-generated out of its local counterpart: **{local_arraytype_name}**.

    The implementation is based on IPython.parallel and local {local_arraytype_name} - arrays.
    Every special operation {local_arraytype_name} implements ("__add__", "__le__", ...) is also
    available for {arraytype_name}.

    Note that array slicing is a cheap operation since no memory is copied.
    However this can easily lead to the situation where you end up with two arrays of the same size but of distinct element distribution.
    Therefore call dist_like() first before doing any manual stuff on their local arrays.
    However every cluster-wide array operation first equalizes the distribution of all involved arrays,
    so an explicit call to dist_like() is rather unlikely in most use cases.

    If you try to access an attribute that is only available for the local array, the request
    is forwarded to an internal local copy of the whole distributed array (see: :meth:`gather()`).
    This internal copy is only created when you want to access it and is held until ``__setitem__`` is called,
    i.e. the array's content is manipulated.
    """
    local_arraytype = None
    target_modulename = None
    interengine_copier = None
    may_allocate = True

    def __init__(self, shape, dtype=np.float, distaxes='all', target_offsets=None, target_ranks=None, no_allocation=False, **kwargs):
        """Creates an instance of {arraytype_name}. This is a low-level method of instantiating an array, it should rather be
        constructed using factory functions ("empty", "zeros", "open", ...)

        :param ints shape: shape of array
        :param dtype: datatype of a single element
        :param ints distaxes: distributed axes. Accepts a single integer too. Defaults to 'all' meaning each axis is distributed.
        :param target_offsets: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains the offsets of the local array.
        :type target_offsets: list of lists
        :param ints target_ranks: linear list of :term:`engine` ranks holding the local arrays.
            The last distributed axis is iterated over first.
        :param bool no_allocation: if ``True`` no instance of {local_arraytype_name} will be created on engine. Useful for
            manual instantiation of the local array.
        :param kwargs: additional keyword arguments are forwarded to the constructor of the local array.
        """
        #: size of the array on each axis
        if type(shape) not in (list, tuple):
            shape = (shape,)
        elif type(shape) is not tuple:
            shape = tuple(shape)
        self.shape = shape
        ##: datatype of a single data value
        self.dtype = dtype
        if distaxes == 'all':
            distaxes = tuple(range(len(shape)))
        elif type(distaxes) not in (list, tuple):
            distaxes = (distaxes,)
        elif type(distaxes) is not tuple:
            distaxes = tuple(distaxes)
        #: axes on which memory is distributed across :term:`engines <engine>`
        self.distaxes = distaxes
        #: total bytes consumed by elements of this array.
        self.nbytes = np.dtype(dtype).itemsize * np.prod(self.shape)
        self.view = com.getView()
        self.kwargs = kwargs
        self.local_copy_is_dirty = False

        assert len(distaxes) <= len(shape),\
            "more distributed axes ({}) than dimensions ({})".format(len(distaxes), len(shape))
        for distaxis in distaxes:
            assert distaxis >= 0 and distaxis < len(self.shape),\
                "distributed axis ({}) has to be within [0,{}]".format(distaxis, len(self.shape)-1)

        if target_offsets is None and target_ranks is None:
            # create hypothetical patch with best surface-to-volume ratio
            patch_volume = np.prod([self.shape[distaxis] for distaxis in distaxes]) / float(len(self.view.targets))
            patch_edge_length = pow(patch_volume, 1.0/len(distaxes))

            def factorize(n):
                if n == 1: yield 1; return
                for f in range(2,n//2+1) + [n]:
                    while n%f == 0:
                        n //= f
                        yield f
            prime_factors = list(factorize(len(self.view.targets)))[::-1] # get prime factors of number of engines in descending order
            sorted_distaxes = sorted(distaxes, key=lambda axis: self.shape[axis]) # sort distributed axes in ascending order
            # calculate number of available targets (engines) per distributed axis
            # This value should be close to array_edge_length / patch_edge_length
            num_targets_av = [1] * len(self.shape)

            for distaxis in sorted_distaxes[:-1]:
                num_patches = self.shape[distaxis] / patch_edge_length
                while float(num_targets_av[distaxis]) < num_patches and prime_factors:
                    num_targets_av[distaxis] *= prime_factors.pop()
            # the largest axis gets the remaining (largest) prime_factors
            if prime_factors:
                num_targets_av[sorted_distaxes[-1]] *= np.prod(prime_factors)

            # calculate target_offsets
            localshape = np.array(self.shape)
            for distaxis in distaxes:
                localshape[distaxis] = (self.shape[distaxis] - 1) / num_targets_av[distaxis] + 1

            # number of occupied targets for each distributed axis by this ndarray instance
            num_targets = [(self.shape[distaxis] - 1) / localshape[distaxis] + 1 for distaxis in distaxes]

            # calculate target_offsets
            target_offsets = [np.arange(num_targets[i]) * localshape[distaxes[i]] for i in range(len(distaxes))]

            # generate target_ranks list
            target_ranks = tuple(range(np.prod(num_targets)))

        if target_offsets is None:
            localshape = np.array(self.shape)
            for distaxis in distaxes:
                localshape[distaxis] = (self.shape[distaxis] - 1) / len(target_ranks[distaxis]) + 1
            target_offsets = [np.arange(num_targets[i]) * localshape[distaxes[i]] for i in range(len(distaxes))]

        if target_ranks is None:
            num_targets = [len(target_offsets_axis) for target_offsets_axis in target_offsets]
            target_ranks = tuple(range(np.prod(num_targets)))
        elif type(target_ranks) is not tuple:
            target_ranks = tuple(target_ranks)

        self.target_offsets = target_offsets
        self.target_ranks = target_ranks

        # generate a unique variable name used on target representing this instance
        global array_id
        #: Unique variable name of the local *array* on *engine*.
        #: Unless you are doing manual stuff on the *engines* there is no need for dealing with this attribute.
        self.name = 'dist_array' + str(array_id)
        array_id += 1

        if no_allocation:
            self.view.push({self.name : None}, targets=self.target_ranks)
        else:
            target_shapes = self.target_shapes()

            self.view.scatter('target_shape', target_shapes, targets=self.target_ranks)
            self.view.push({'kwargs' : kwargs, 'dtype' : dtype}, targets=self.target_ranks)
            self.view.execute('%s = %s(shape=target_shape[0], dtype=dtype, **kwargs)' % \
                (self.name, self.__class__.target_modulename + "." + self.__class__.local_arraytype.__name__), targets=self.target_ranks)

    def __del__(self):
        self.view.execute('del %s' % self.name, targets=self.target_ranks)

    def target_shapes(self):
        """generate a list of the local shape on each target in use"""
        targetshapes = []
        num_targets = [len(target_offsets_axis) for target_offsets_axis in self.target_offsets]
        for rank_idx_vector in np.ndindex(*num_targets): # last dimension is iterated over first
            targetshape = list(self.shape)
            for distaxis_idx in range(len(self.distaxes)):
                distaxis = self.distaxes[distaxis_idx]
                i = rank_idx_vector[distaxis_idx]
                end = self.target_offsets[distaxis_idx][i+1] if i != num_targets[distaxis_idx] - 1 else self.shape[distaxis]
                targetshape[distaxis] = end - self.target_offsets[distaxis_idx][i]
            targetshapes.append(targetshape)

        return targetshapes

    def target_offset_vectors(self):
        """generate a list of the local offset vectors on each target in use"""
        target_offset_vectors = []
        num_targets = [len(target_offsets_axis) for target_offsets_axis in self.target_offsets]
        for rank_idx_vector in np.ndindex(*num_targets): # last dimension is iterated over first
            target_offset_vector = [0] * len(self.shape)
            for distaxis_idx in range(len(self.distaxes)):
                distaxis = self.distaxes[distaxis_idx]
                i = rank_idx_vector[distaxis_idx]
                target_offset_vector[distaxis] = self.target_offsets[distaxis_idx][i]
            target_offset_vectors.append(target_offset_vector)

        return target_offset_vectors

    def __get_linear_rank_idx(self, rank_idx_vector):
        # convert rank_idx_vector to linear rank index
        # last dimension is iterated over first
        num_targets = [len(target_offsets_axis) for target_offsets_axis in self.target_offsets]
        pitch = [int(np.prod(num_targets[i+1:])) for i in range(len(self.distaxes))]
        return sum(rank_idx_component * pitch_component for rank_idx_component, pitch_component in zip(rank_idx_vector, pitch))

    def __getitem__(self, args):
        # bitmask indexing
        if isinstance(args, self.__class__) and args.dtype == bool:
            bitmask = args
            assert bitmask.shape == self.shape,\
                "shape of bitmask (%s) does not correspond to shape of array (%s)"\
                    % (str(bitmask.shape), str(self.shape))

            bitmask = bitmask.dist_like(self) # equalize distribution if necessary
            self.view.execute("tmp = {0}[{1}]; tmp_size = tmp.shape[0]".format(repr(self), repr(bitmask)), targets=self.target_ranks)
            sizes = self.view.pull("tmp_size", targets=self.target_ranks)
            new_target_ranks = [rank for rank, size in zip(self.target_ranks, sizes) if size > 0]
            new_sizes = [size for size in sizes if size > 0]
            partial_sum = lambda a, b: a + [a[-1] + b]
            new_target_offsets = [ [0] + reduce(partial_sum, new_sizes[1:-1], new_sizes[0:1]) ]
            new_shape = [sum(new_sizes)]
            # create resulting ndarray
            result = self.__class__(new_shape, self.dtype, 0, new_target_offsets, new_target_ranks, no_allocation=True, **self.kwargs)
            self.view.execute("{0} = tmp; del tmp".format(result.name), targets=result.target_ranks)
            return result

        if args == slice(None):
            args = (slice(None),) * len(self.shape)

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = (args,)

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(args), len(self.shape))

        # wrap all integer indices
        args = [(arg + s) % s if type(arg) is int else arg for arg, s in zip(args, self.shape)]

        # shape of the new sliced ndarray
        new_shape, clean_view = helper.view_of_shape(self.shape, args)

        # if args is a list of indices then return a single data value
        if not new_shape:
            local_idx = list(args)
            rank_idx_vector = []
            for distaxis, target_offsets in zip(self.distaxes, self.target_offsets):
                dist_idx = args[distaxis]
                rank_idx_component = np.searchsorted(target_offsets, dist_idx, side="right") - 1
                local_idx[distaxis] = dist_idx - target_offsets[rank_idx_component]
                rank_idx_vector.append(rank_idx_component)

            rank_idx = self.__get_linear_rank_idx(rank_idx_vector)
            return self.view.pull("%s%s" % (self.name, repr(local_idx)), targets=self.target_ranks[rank_idx])

        if all(type(clean_view[distaxis]) is int for distaxis in self.distaxes):
            # return local array because all distributed axes have vanished
            rank_idx_vector = []
            for distaxis, target_offsets in zip(self.distaxes, self.target_offsets):
                dist_idx = clean_view[distaxis]
                rank_idx_component = np.searchsorted(target_offsets, dist_idx, side="right") - 1
                clean_view[distaxis] = dist_idx - target_offsets[rank_idx_component]
                rank_idx_vector.append(rank_idx_component)

            rank_idx = self.__get_linear_rank_idx(rank_idx_vector)
            self.view.execute("sliced = %s%s" % (self.name, repr(clean_view)), targets=self.target_ranks[rank_idx])
            return self.view.pull("sliced", targets=self.target_ranks[rank_idx])

        # determine properties of the new, sliced ndarray
        # keep these in mind when reading the following for loop
        local_slices_aa = [] # aa = all (distributed) axes
        new_target_offsets = []
        new_rank_ids_aa = []
        new_distaxes = []

        for distaxis, distaxis_idx, target_offsets \
            in zip(self.distaxes, range(len(self.distaxes)), self.target_offsets):

            # slice object in the direction of the distributed axis
            distaxis_slice = clean_view[distaxis]

            # if it turns out that distaxis_slice is actually an int, this axis has to vanish.
            # That means the local slice object has to be an int too.
            if type(distaxis_slice) == int:
                dist_idx = distaxis_slice
                rank_idx_component = np.searchsorted(target_offsets, dist_idx, side="right") - 1
                local_idx = dist_idx - target_offsets[rank_idx_component]
                local_slices_aa.append((local_idx,))
                new_rank_ids_aa.append((rank_idx_component,))
                continue

            # determine properties of the new, sliced ndarray
            local_slices_sa = [] # sa = single axis
            new_target_offsets_sa = []
            new_rank_ids_sa = []
            total_ids = 0

            first_rank_idx = np.searchsorted(target_offsets, distaxis_slice.start, side="right") - 1
            last_rank_idx = np.searchsorted(target_offsets, distaxis_slice.stop, side="right")

            for i in range(first_rank_idx, last_rank_idx):

                # index range of current target
                begin = target_offsets[i]
                end = target_offsets[i+1] if i < len(target_offsets)-1 else self.shape[distaxis]
                # first slice index within [begin, end)
                firstSliceIdx = helper.getFirstSliceIdx(distaxis_slice, begin, end)
                if firstSliceIdx is None: continue
                # calculate last slice index of distaxis_slice
                tmp = (distaxis_slice.stop-1 - distaxis_slice.start) / distaxis_slice.step
                lastIdx = distaxis_slice.start + tmp * distaxis_slice.step
                # calculate last sub index within [begin,end)
                tmp = (end-1 - firstSliceIdx) / distaxis_slice.step
                lastSliceIdx = firstSliceIdx + tmp * distaxis_slice.step
                lastSliceIdx = min(lastSliceIdx, lastIdx)
                # slice object for current target
                local_slices_sa.append(slice(firstSliceIdx - begin, lastSliceIdx+1 - begin, distaxis_slice.step))
                # number of indices remaining on the current target after slicing
                num_ids = (lastSliceIdx - firstSliceIdx) / distaxis_slice.step + 1
                # new offset for current target
                new_target_offsets_sa.append(total_ids)
                total_ids += num_ids
                # target rank index
                new_rank_ids_sa.append(i)

            new_rank_ids_sa = np.array(new_rank_ids_sa)

            local_slices_aa.append(local_slices_sa)
            new_target_offsets.append(new_target_offsets_sa)
            new_rank_ids_aa.append(new_rank_ids_sa)

            # shift distaxis by the number of vanished axes in the left of it
            new_distaxis = distaxis - sum(1 for arg in args[:distaxis] if type(arg) is int)
            new_distaxes.append(new_distaxis)

        # create list of targets which participate slicing, new_target_ranks
        num_ranks_aa = [len(new_rank_ids_sa) for new_rank_ids_sa in new_rank_ids_aa]
        new_target_ranks = []
        for idx in np.ndindex(*num_ranks_aa):
            rank_idx_vector = [new_rank_ids_aa[i][idx[i]] for i in range(len(idx))]
            rank_idx = self.__get_linear_rank_idx(rank_idx_vector)
            new_target_ranks.append(self.target_ranks[rank_idx])

        # create resulting ndarray
        result = self.__class__(new_shape, self.dtype, new_distaxes, new_target_offsets, new_target_ranks, no_allocation=True, **self.kwargs)

        # remote slicing
        local_args_list = []
        for idx in np.ndindex(*num_ranks_aa):
            local_slices = [local_slices_aa[i][idx[i]] for i in range(len(idx))]
            local_args = list(args)
            for distaxis, distaxis_idx in zip(self.distaxes, range(len(self.distaxes))):
                local_args[distaxis] = local_slices[distaxis_idx]
            local_args_list.append(local_args)

        self.view.scatter('local_args', local_args_list, targets=result.target_ranks)
        self.view.execute('%s = %s[local_args[0]]' % (result.name, self.name), targets=result.target_ranks)

        return result

    def __setitem__(self, key, value):
        self.local_copy_is_dirty = True
        # bitmask indexing
        if isinstance(key, self.__class__) and key.dtype == bool:
            bitmask = key.dist_like(self)
            self.view.execute("%s[%s] = %s" % (repr(self), repr(bitmask), repr(value)), targets=self.target_ranks)
            return

        # if args is [:] then assign value to the entire ndarray
        if key == slice(None):
            # assign local array to self
            if isinstance(value, self.__class__.local_arraytype):
                subarrays = []
                for target_offset_vector, target_shape in zip(self.target_offset_vectors(), self.target_shapes()):
                    window = [slice(start, start+length) for start, length in zip(target_offset_vector, target_shape)]
                    subarrays.append(value[window])

                self.view.scatter("subarray", subarrays, targets=self.target_ranks)
                self.view.execute("%s[:] = subarray[0]" % self.name, targets=self.target_ranks)
                return

            # assign other array or value to self
            other = value.dist_like(self) if hasattr(value, "dist_like") else value
            self.view.execute("%s[:] = %s" % (repr(self), repr(other)), targets=self.target_ranks)
            return

        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        assert len(key) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(key), len(self.shape))

        # value assignment (key == list of indices)
        if all(type(k) is int for k in key):
            local_idx = list(key)
            local_idx = [(i + s) % s for i, s in zip(local_idx, self.shape)]
            rank_idx_vector = []
            for distaxis, target_offsets in zip(self.distaxes, self.target_offsets):
                dist_idx = key[distaxis]
                rank_idx_component = np.searchsorted(target_offsets, dist_idx, side="right") - 1
                local_idx[distaxis] = dist_idx - target_offsets[rank_idx_component]
                rank_idx_vector.append(rank_idx_component)

            rank_idx = self.__get_linear_rank_idx(rank_idx_vector)
            self.view.push({'value' : value}, targets=self.target_ranks[rank_idx])
            self.view.execute("%s%s = value" % (self.name, repr(local_idx)), targets=self.target_ranks[rank_idx])
            return

        # assign value to sub-array of self
        sub_array = self[key]
        sub_array[:] = value

    def __str__(self):
        return self.gather().__str__()

    def __repr__(self):
        return self.name

    @property
    def local_copy(self):
        if not hasattr(self, "_local_copy") or self.local_copy_is_dirty:
            self._local_copy = self.gather()
            self.local_copy_is_dirty = False
        return self._local_copy

    def __getattr__(self, name):
        """If the requested attribute is an attribute of the local array and not of this array then
        gather() is called and the request is forwarded to the gathered array. This makes this
        distributed array more behaving like a local array."""
        if not hasattr(self.__class__.local_arraytype, name):
            raise AttributeError(name)

        return getattr(self.local_copy, name)

    def gather(self):
        """Gathers local instances of {local_arraytype_name} from *engines*, concatenates them and returns
        the result.

        .. note:: You may not call this method explicitly because if you try to access an attribute
            of the local array ({local_arraytype_name}), ``gather()`` is called implicitly before the request is forwarded
            to that internal gathered array. Just access attributes like you do for the local array.
            The internal copy is held until ``__setitem__`` is called, e.g. ``a[1] = 3.0``, setting
            a dirty flag to the local copy.

        .. warning:: If another array overlapping this array is manipulating its data there is no chance to set
            the dirty flag so you have to keep in mind to call ``gather()`` explicitly in this case!

        :return: instance of {local_arraytype_name}
        """
        local_arrays = self.view.pull(self.name, targets=self.target_ranks)

        result = self.__class__.local_arraytype(shape=self.shape, dtype=self.dtype, **self.kwargs)

        for target_offset_vector, target_shape, local_array \
            in zip(self.target_offset_vectors(), self.target_shapes(), local_arrays):

            window = [slice(start, start+length) for start, length in zip(target_offset_vector, target_shape)]
            result[window] = local_array

        return result

    def copy(self):
        """Returns a hard copy of this array.
        """
        assert self.__class__.may_allocate == True, "{0} is not allowed to allocate new memory.".format(self.__class__.__name__)

        result = self.__class__(self.shape, self.dtype, self.distaxes, self.target_offsets, self.target_ranks, no_allocation=True, **self.kwargs)
        self.view.execute("%s = %s.copy()" % (result.name, self.name), targets=self.target_ranks)
        return result

    def is_distributed_like(self, other):
        if self.distaxes == other.distaxes:
          if all(np.array_equal(my_target_offsets, other_target_offsets) \
            for my_target_offsets, other_target_offsets in zip(self.target_offsets, other.target_offsets)):
              if self.target_ranks == other.target_ranks:
                  return True
        return False

    def dist_like(self, other):
        """Redistributes a copy of this array (*self*) like *other* and returns the result.
        Checks whether redistribution is necessary and returns *self* if not.

        Redistribution involves inter-engine communication.

        :param other: target array
        :type other: distributed array
        :raises AssertionError: if the shapes of *self* and *other* don't match.
        :return: new array with the same content as *self* but distributed like *other*.
            If *self* is already distributed like *other* nothing is done and *self* is returned.
        """
        assert self.shape == other.shape,\
            "Shapes do not match: " + str(self.shape) + " <-> " + str(other.shape)

        # if self is already distributed like *other* do nothing
        if self.is_distributed_like(other):
            return self

        assert self.__class__.may_allocate, "{0} is not allowed to allocate new memory.".format(self.__class__.__name__)

        # todo: optimization -> helper.common_decomposition should be a generator
        common_axes, common_offsets, common_idx_pairs = \
            helper.common_decomposition(self.distaxes, self.target_offsets, other.distaxes, other.target_offsets, self.shape)

        my_commData = defaultdict(list)
        other_commData = defaultdict(list)

        tag = 0
        num_offsets = [len(common_offsets_sa) for common_offsets_sa in common_offsets]

        for idx in np.ndindex(*num_offsets):
            my_rank_idx_vector = [0] * len(self.distaxes)
            other_rank_idx_vector = [0] * len(other.distaxes)
            my_window = [slice(None)] * len(self.shape)
            other_window = [slice(None)] * len(self.shape)

            for i in range(len(idx)):
                common_axis = common_axes[i]
                begin = common_offsets[i][idx[i]]
                end = common_offsets[i][idx[i]+1] if idx[i] < len(common_offsets[i]) -1 else self.shape[common_axes[i]]

                if common_axis in self.distaxes and common_axis in other.distaxes:
                    my_rank_idx_comp = common_idx_pairs[i][idx[i],0]
                    other_rank_idx_comp = common_idx_pairs[i][idx[i],1]
                    my_rank_idx_vector[self.distaxes.index(common_axis)] = my_rank_idx_comp
                    other_rank_idx_vector[other.distaxes.index(common_axis)] = other_rank_idx_comp

                    my_offset = self.target_offsets[self.distaxes.index(common_axis)][my_rank_idx_comp]
                    my_window[common_axis] = slice(begin - my_offset, end - my_offset)
                    other_offset = other.target_offsets[other.distaxes.index(common_axis)][other_rank_idx_comp]
                    other_window[common_axis] = slice(begin - other_offset, end - other_offset)

                    continue

                if common_axis in self.distaxes:
                    my_rank_idx_vector[self.distaxes.index(common_axis)] = idx[i]
                    other_window[common_axis] = slice(begin, end)
                if common_axis in other.distaxes:
                    other_rank_idx_vector[other.distaxes.index(common_axis)] = idx[i]
                    my_window[common_axis] = slice(begin, end)

            my_rank = self.target_ranks[self.__get_linear_rank_idx(my_rank_idx_vector)]
            other_rank = other.target_ranks[other.__get_linear_rank_idx(other_rank_idx_vector)]

            my_commData[my_rank].append((other_rank, my_window, tag))
            other_commData[other_rank].append((my_rank, other_window, tag))

            tag += 1

        # push communication meta-data to engines
        self.view.scatter('src_commData', my_commData.values(), targets=my_commData.keys())
        self.view.scatter('dest_commData', other_commData.values(), targets=other_commData.keys())

        # result ndarray
        result = self.__class__(self.shape, self.dtype, other.distaxes, other.target_offsets, other.target_ranks, False, **self.kwargs)

        self.__class__.interengine_copier(self, result)

        return result

    def info(self, name):
        print name + " info:"
        print "{}.name".format(name), self.name
        print "{}.target_ranks".format(name), self.target_ranks
        print "{}.target_offsets".format(name), self.target_offsets
        print "{}.distaxes".format(name), self.distaxes
        self.view.execute("dt = str({}.dtype)".format(repr(self)), targets=self.target_ranks)
        print "{}.dtypes".format(name), self.view.pull("dt", targets=self.target_ranks)
        self.view.execute("t = str(type({}))".format(repr(self)), targets=self.target_ranks)
        print "{}.types".format(name), self.view.pull("t", targets=self.target_ranks)

    def __elementwise_op__(self, op, *args):
        args = [arg.dist_like(self) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)

        result = self.__class__(self.shape, self.dtype, self.distaxes, self.target_offsets, self.target_ranks, no_allocation=True, **self.kwargs)

        self.view.execute("{0} = {1}.{2}({3}); dtype={0}.dtype".format(repr(result), repr(self), op, arg_string), targets=self.target_ranks)
        result.dtype = self.view.pull("dtype", targets=result.target_ranks[0])
        result.nbytes = np.dtype(result.dtype).itemsize * np.prod(result.shape)

        return result

    def __elementwise_iop__(self, op, *args):
        args = [arg.dist_like(self) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)
        self.view.execute("%s = %s.%s(%s)" % (repr(self), repr(self), op, arg_string), targets=self.target_ranks)
        return self

#----------------------------------------------------------------

from multiple_axes_funcs import *