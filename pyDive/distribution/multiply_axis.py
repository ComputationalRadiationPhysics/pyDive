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

array_id = 0

class DistributedGenericArray(object):
    """
    Represents a cluster-wide, multidimensional, homogeneous array of fixed-size elements.
    *cluster-wide* means that its elements are distributed across IPython.parallel-engines.
    The distribution is done in one or multiply dimension along user-specified axes.
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
    """
    local_arraytype = None
    target_modulename = None
    interengine_copier = None
    may_allocate = True

    def __init__(self, shape, dtype=np.float, distaxes=0, target_offsets=None, target_ranks=None, no_allocation=False, **kwargs):
        """Creates an instance of {arraytype_name}. This is a low-level method of instantiating an array, it should rather be
        constructed using factory functions ("empty", "zeros", "open", ...)

        :param ints shape: shape of array
        :param dtype: datatype of a single element
        :param ints distaxes: distributed axes. Accepts a single integer too.
        :param target_offsets: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains of the offsets of the local array.
        :type target_offsets: list of lists
        :param ints target_ranks: List of :term:`engine` ranks holding the local arrays.
        :param bool no_allocation: if ``True`` no instance of {local_arraytype_name} will be created on engine. Useful for
            manual instantiation of the local array.
        :param kwargs: additional keyword arguments are forwarded to the constructor of the local array.
        """
        #: size of the array on each axis
        self.shape = tuple(shape)
        ##: datatype of a single data value
        self.dtype = dtype
        #: axes on which memory is distributed across the :term:`engines <engine>`
        if type(distaxes) not in (list, tuple):
            distaxes = tuple(distaxes)
        self.distaxes = distaxes
        ##: total bytes consumed by the elements of the array.
        self.nbytes = np.dtype(dtype).itemsize * np.prod(self.shape)
        self.view = com.getView()
        self.kwargs = kwargs

        assert len(distaxes) <= len(shape),\
            "more distributed axes {} than dimensions {}".format(len(distaxes), len(shape))
        for distaxis in distaxes:
            assert distaxis >= 0 and distaxis < len(self.shape),\
                "distributed axis ({}) has to be within [0,{})".format(distaxis, len(self.shape))

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
                while float(num_targets_av[distaxis]) <= num_patches and prime_factors:
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
            target_offsets = [np.arange(num_targets[distaxis]) * localshape[distaxis] for distaxis in distaxes]

#            pitch = [np.prod(num_targets[:i]) for i in len(distaxes)]

            # generate target_ranks list
            target_ranks = tuple(range(np.prod(num_targets)))

        if target_offsets is None:
            localshape = np.array(self.shape)
            for distaxis in distaxes:
                localshape[distaxis] = (self.shape[distaxis] - 1) / len(target_ranks[distaxis]) + 1
            target_offsets = [np.arange(num_targets[distaxis]) * localshape[distaxis] for distaxis in distaxes]

        if target_ranks is None:
            num_targets = [len(target_offsets[distaxis]) for distaxis in distaxes]
            target_ranks = tuple(range(np.prod(num_targets)))

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
        # generate a list of the local shape on each target in use
        targetshapes = []
        num_targets = [len(target_offsets[distaxis]) for distaxis in distaxes]
        for idx in np.ndindex(num_targets):
            targetshape = np.array(self.shape)
            for distaxis in self.distaxes:
                i = idx[distaxis]
                end = self.target_offsets[distaxis][i+1] if i != num_targets[distaxis] - 1 else num_targets[distaxis]
                targetshape[distaxis] = end - self.target_offsets[distaxis][i]
            targetshapes.append(targetshape)

        return targetshapes

    def target_offset_vectors(self):
        # generate a list of the local offset vectors on each target in use
        target_offset_vectors = []
        num_targets = [len(target_offsets[distaxis]) for distaxis in distaxes]
        for idx in np.ndindex(num_targets):
            target_offset_vector = [0] * len(self.shape)
            for distaxis in self.distaxes:
                target_offset_vector[distaxis] = self.target_offsets[distaxis][idx[distaxis]]
            target_offset_vectors.append(target_offset_vector)

        return target_offset_vectors

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
            new_target_offsets = [0] + reduce(partial_sum, new_sizes[1:-1], new_sizes[0:1])
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

        # shape of the new sliced ndarray
        new_shape, clean_view = helper.view_of_shape(self.shape, args)

        # if args is a list of indices then return a single data value
        if not new_shape:
            dist_idx = args[self.distaxis]
            target_idx = np.searchsorted(self.target_offsets, dist_idx, side="right") - 1
            local_idx = list(args)
            local_idx[self.distaxis] = dist_idx - self.target_offsets[target_idx]
            return self.view.pull("%s%s" % (self.name, repr(local_idx)), targets=self.target_ranks[target_idx])

        if type(clean_view[self.distaxis]) is int:
            # return local array because the distributed axis has vanished
            dist_idx = clean_view[self.distaxis]
            target_idx = np.searchsorted(self.target_offsets, dist_idx, side="right") - 1
            clean_view[self.distaxis] = dist_idx - self.target_offsets[target_idx]
            self.view.execute("sliced = %s%s" % (self.name, repr(clean_view)), targets=self.target_ranks[target_idx])
            return self.view.pull("sliced", targets=self.target_ranks[target_idx])

        # slice object in the direction of the distributed axis
        distaxis_slice = clean_view[self.distaxis]

        # determine properties of the new sliced ndarray
        new_target_offsets = []
        new_target_ranks = []
        local_slices = []
        total_ids = 0

        first_target_idx = np.searchsorted(self.target_offsets, distaxis_slice.start, side="right") - 1
        last_target_idx = np.searchsorted(self.target_offsets, distaxis_slice.stop, side="right")

        for i, target in zip(range(first_target_idx, last_target_idx),\
                             self.target_ranks[first_target_idx:last_target_idx]):
            # index range of current target
            begin = self.target_offsets[i]
            end = self.target_offsets[i+1] if i+1 < len(self.target_offsets) else self.shape[self.distaxis]
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
            local_slices.append(slice(firstSliceIdx - begin, lastSliceIdx+1 - begin, distaxis_slice.step))
            # number of indices remaining on the current target after slicing
            num_ids = (lastSliceIdx - firstSliceIdx) / distaxis_slice.step + 1
            # new offset for current target
            new_target_offsets.append(total_ids)
            total_ids += num_ids
            # target rank
            new_target_ranks.append(target)

        # shift distaxis by the number of vanished axes in the left of it
        new_distaxis = self.distaxis - sum(1 for arg in args[:self.distaxis] if type(arg) is int)

        # create resulting ndarray
        result = self.__class__(new_shape, self.dtype, new_distaxis, new_target_offsets, new_target_ranks, no_allocation=True, **self.kwargs)

        # remote slicing
        local_args_list = []
        for local_slice in local_slices:
            local_args = list(args)
            local_args[self.distaxis] = local_slice
            local_args_list.append(local_args)

        self.view.scatter('local_args', local_args_list, targets=result.target_ranks)
        self.view.execute('%s = %s[local_args[0]]' % (result.name, self.name), targets=result.target_ranks)

        return result

    def __setitem__(self, key, value):
        # bitmask indexing
        if isinstance(key, self.__class__) and key.dtype == bool:
            bitmask = key.dist_like(self)
            self.view.execute("%s[%s] = %s" % (repr(self), repr(bitmask), repr(value)), targets=self.target_ranks)
            return

        # if args is [:] then assign value to the entire ndarray
        if key == slice(None):
            # assign local array to self
            if isinstance(value, self.__class__.local_arraytype):
                window = [slice(None)] * len(self.shape)
                subarrays = []
                for i in range(len(self.target_offsets)):
                    begin = self.target_offsets[i]
                    end = self.target_offsets[i+1] if i+1 < len(self.target_offsets) else self.shape[self.distaxis]
                    window[self.distaxis] = slice(begin, end)
                    subarrays.append(value[window])
                self.view.scatter("subarray", subarrays, targets=self.target_ranks)
                self.view.execute("%s[:] = subarray[0]" % self.name, targets=self.target_ranks)
                return

            # assign other array to self
            if isinstance(value, self.__class__):
                other = value.dist_like(self)
                self.view.execute("%s[:] = %s" % (self.name, other.name), targets=self.target_ranks)
                return

            # assign single value to self
            self.view.execute("%s[:] = %s" % (self.name, repr(value)), targets=self.target_ranks)
            return

        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        assert len(key) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(key), len(self.shape))

        # value assignment (key == list of indices)
        if all(type(k) is int for k in key):
            dist_idx = key[self.distaxis]
            target_idx = np.searchsorted(self.target_offsets, dist_idx, side="right") - 1
            local_idx = list(key)
            local_idx[self.distaxis] = dist_idx - self.target_offsets[target_idx]
            self.view.push({'value' : value}, targets=self.target_ranks[target_idx])
            self.view.execute("%s%s = value" % (self.name, repr(local_idx)), targets=self.target_ranks[target_idx])
            return

        # assign value to sub-array of self
        sub_array = self[key]
        sub_array[:] = value

    def __str__(self):
        return self.gather().__str__()

    def __repr__(self):
        return self.name

    def gather(self):
        """Gathers local instances of {local_arraytype_name} from *engines*, concatenates them and returns
        the result.

        :return: instance of {local_arraytype_name}
        """
        local_arrays = self.view.pull(self.name, targets=self.target_ranks)

        result = self.__class__.local_arraytype(shape=self.shape, dtype=self.dtype, **self.kwargs)
        window = [slice(None)] * len(self.shape)
        for i, local_array in zip(range(len(self.target_offsets)), local_arrays):
            begin = self.target_offsets[i]
            end = self.target_offsets[i+1] if i+1 < len(self.target_offsets) else self.shape[self.distaxis]
            window[self.distaxis] = slice(begin, end)
            result[window] = local_array

        return result

    def copy(self):
        """Returns a hard copy of this array.
        """
        assert self.__class__.may_allocate == True, "{0} is not allowed to allocate new memory.".format(self.__class__.__name__)

        result = self.__class__(self.shape, self.dtype, self.distaxis, self.target_offsets, self.target_ranks, no_allocation=True, **self.kwargs)
        self.view.execute("%s = %s.copy()" % (result.name, self.name), targets=self.target_ranks)
        return result

    def dist_like(self, other):
        """Redistributes a copy of this array (*self*) like *other* and returns the result.
        Checks whether redistribution is necessary and returns *self* if not.

        Redistribution involves inter-engine communication.

        :param other: target array
        :type other: distributed array
        :raises AssertionError: if the shapes of *self* and *other* don't match.
        :raises AssertionError: if *self* and *other* are distributed along distinct axes.
        :return: new array with the same content as *self* but distributed like *other*.
            If *self* is already distributed like *other* nothing is done and *self* is returned.
        """
        assert self.shape == other.shape,\
            "Shapes do not match: " + str(self.shape) + " <-> " + str(other.shape)
        assert self.distaxis == other.distaxis # todo: add this feature

        # if self is already distributed like *other* do nothing
        if np.array_equal(self.target_offsets, other.target_offsets):
            if self.target_ranks == other.target_ranks:
                return self

        assert self.__class__.may_allocate, "{0} is not allowed to allocate new memory.".format(self.__class__.__name__)

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
            end_self = self.target_offsets[self_idx+1] if self_idx+1 < len(self.target_offsets) else self.shape[self.distaxis]
            end_other = other.target_offsets[other_idx+1] if other_idx+1 < len(other.target_offsets) else other.shape[other.distaxis]
            end = min(end_self, end_other)
            partition_size = end - begin

            # source's meta-data
            src_targets[self_idx].append(other.target_ranks[other_idx])
            src_tags[self_idx].append(tag)
            src_distaxis_sizes[self_idx].append(partition_size)

            # destination's meta-data
            dest_targets[other_idx].append(self.target_ranks[self_idx])
            dest_tags[other_idx].append(tag)
            dest_distaxis_sizes[other_idx].append(partition_size)

            # go to the next common partition
            if end == end_self:
                if not end == self.shape[self.distaxis]:
                    src_targets.append([])
                    src_tags.append([])
                    src_distaxis_sizes.append([])
                self_idx += 1
            if end == end_other:
                if not end == self.shape[self.distaxis]:
                    dest_targets.append([])
                    dest_tags.append([])
                    dest_distaxis_sizes.append([])
                other_idx += 1

            begin = end
            tag += 1

        # push communication meta-data to engines
        self.view.scatter('src_targets', src_targets, targets=self.target_ranks)
        self.view.scatter('src_tags', src_tags, targets=self.target_ranks)
        self.view.scatter('src_distaxis_sizes', src_distaxis_sizes, targets=self.target_ranks)

        self.view.scatter('dest_targets', dest_targets, targets=other.target_ranks)
        self.view.scatter('dest_tags', dest_tags, targets=other.target_ranks)
        self.view.scatter('dest_distaxis_sizes', dest_distaxis_sizes, targets=other.target_ranks)

        # result ndarray
        result = self.__class__(self.shape, self.dtype, self.distaxis, other.target_offsets, other.target_ranks, False, **self.kwargs)

        self.__class__.interengine_copier(self, result)

        return result

    def __elementwise_op__(self, op, *args):
        args = [arg.dist_like(self) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)

        result = self.__class__(self.shape, self.dtype, self.distaxis, self.target_offsets, self.target_ranks, no_allocation=True, **self.kwargs)

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

#from single_axis_funcs import *