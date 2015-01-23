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
import helper as helper
import factories as factories
from .. import IPParallelClient as com
import dist_math
import sys

ndarray_id = 0

class ndarray(object):
    """Represents a cluster-wide, multidimensional, homogenous array of fixed-size elements.
    *cluster-wide* means that its elements are distributed across :term:`IPython.parallel-engines <engine>`.
    The distribution is done in one dimension along a user-specified axis. The user can optionally specify
    which engine maps to which index range or leave the default that persuits an uniform distribution
    across all engines.

    The implementation is based on *IPython.parallel* and local *numpy-arrays*. The design goal is
    to forward every *numpy-array* method onto the cluster-wide level. Currently :class:`pyDive.ndarray.ndarray.ndarray`
    supports basic arithmetic operations (+ - * / ** //) as well as most of the numpy-*math*
    functions like sin, cos, abs, sqrt, ... (see :mod:`pyDive.ndarray.dist_math`)

    Note that array slicing is a cheap operation since no memory is copied. However this can easily
    lead to the situation where you end up with two arrays of the same size but of distinct element distribution.
    Therefore call :meth:`dist_like` first before doing any manual
    stuff on their local *numpy-arrays*.

    Every cluster-wide array operation first equalizes the distribution of all involved arrays if necessary.
    """
    def __init__(self, shape, distaxis=0, dtype=np.float, target_offsets=None, target_ranks=None, no_allocation=False):
        """Creates an :class:`pyDive.ndarray.ndarray.ndarray` instance. This is a low-level method for instanciating
        an array. Arrays should  be constructed using 'empty', 'zeros'
        or 'array' (see :mod:`pyDive.ndarray.factories`).

        :param shape: size of the array on each axis
        :type shape: tuple of ints
        :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
        :param numpy-dtype dtype: datatype of a single data value
        :param target_offsets: list of indices indicating the offset, along the distributed axis, the local array of
             the corresponding :term:`engine` (see :ref:`target_ranks`) starts with.
        :type target_offsets: *numpy*-array or list/tuple of ints
        :param target_ranks: list of :term:`engine`-ids that share this array.
        :type target_ranks: list/tuple of ints
        :param bool no_allocation: if ``True`` no actual memory, i.e. *numpy-array*, will be
            allocated on :term:`engine`. Useful when you want to assign an existing numpy array manually.

        If *target_offsets* is ``None`` it will be auto-generated
        so that the memory is equally distributed across all :term:`engines <engine>` at its best.
        This means that the last engine may get less memory than the others.
        """

        #: size of the array on each axis
        self.shape = list(shape)
        #: datatype of a single data value
        self.dtype = dtype
        #: axis on which memory is distributed across the :term:`engines <engine>`
        self.distaxis = distaxis
        #: total bytes consumed by the elements of the array.
        self.nbytes = np.dtype(dtype).itemsize * np.prod(self.shape)
        self.view = com.getView()
        self.arraytype = self.__class__

        assert distaxis >= 0 and distaxis < len(self.shape),\
            "distaxis ({}) has to be within [0,{})".format(distaxis, 0, len(self.shape))

        if target_offsets is None or target_ranks is None:
            # number of available targets (engines)
            num_targets_av = len(self.view.targets if target_ranks is None else target_ranks)

            # shape of the local ndarray
            localshape = np.array(self.shape)
            localshape[distaxis] = (self.shape[distaxis] - 1) / num_targets_av + 1

            # number of occupied targets by this ndarray instance
            num_targets = (self.shape[distaxis] - 1) / localshape[distaxis] + 1

        if target_offsets is None:
            #: this is the decomposition of the distributed axis
            self.target_offsets = np.arange(num_targets) * localshape[distaxis]
        else:
            self.target_offsets = np.array(target_offsets)

        if target_ranks is None:
            #: list of indices of the occupied engines
            self.target_ranks = tuple(range(num_targets))
        else:
            self.target_ranks = tuple(target_ranks)

        # generate a unique variable name used on target representing this instance
        global ndarray_id
        #: Unique variable name of the local *numpy-array* on *engine*.
        #: Unless you are doing manual stuff on the *engines* there is no need for dealing with this attribute.
        self.name = 'dist_ndarray' + str(ndarray_id)
        ndarray_id += 1

        if no_allocation:
            self.view.push({self.name : None}, targets=self.target_ranks)
        else:
            # generate a list of the local shape on each target in use
            localshapes = []
            for i in range(len(self.target_offsets)-1):
                localshape = np.array(self.shape)
                localshape[distaxis] = self.target_offsets[i+1] - self.target_offsets[i]
                localshapes.append(localshape)
            localshape = np.array(self.shape)
            localshape[distaxis] = self.shape[distaxis] - self.target_offsets[-1]
            localshapes.append(localshape)

            self.view.scatter('localshape', localshapes, targets=self.target_ranks)
            self.view.push({'dtype' : dtype}, targets=self.target_ranks)
            self.view.execute('%s = np.empty(localshape[0], dtype=dtype)' % self.name, targets=self.target_ranks)

    def __del__(self):
        self.view.execute('del %s' % self.name, targets=self.target_ranks)

    def __getitem__(self, args):
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
            # return numpy-array because the distributed axis has vanished
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
        result = ndarray(new_shape, new_distaxis, self.dtype, new_target_offsets, new_target_ranks, True)

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
        # if args is [:] then assign value to the entire ndarray
        if key == slice(None):
            if isinstance(value, np.ndarray):
                value = factories.array(value, self.distaxis)
            other = value.dist_like(self)
            self.view.execute("%s[:] = %s" % (self.name, other.name), targets=self.target_ranks)
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
        """Gathers the local *numpy-arrays* from the *engines*, concatenates them and returns
        the result.

        :return: numpy-array
        """
        local_arrays = self.view.pull(self.name, targets=self.target_ranks)
        return np.concatenate(local_arrays, axis=self.distaxis)

    def copy(self):
        """Returns a hard copy of this array.
        """
        result = ndarray(self.shape, self.distaxis, self.dtype, self.target_offsets, self.target_ranks, True)
        self.view.execute("%s = %s.copy()" % (result.name, self.name), targets=self.target_ranks)
        return result

    def dist_like(self, other):
        """Redistributes a copy of this array (*self*) like *other* and returns the result.
        Checks whether redistribution is necessary and returns *self* if not.

        Redistribution involves inter-engine communication via MPI.

        :param other: target array
        :type other: :class:`pyDive.ndarray.ndarray.ndarray`
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

        # push communication meta-data to the targets
        self.view.scatter('src_targets', src_targets, targets=self.target_ranks)
        self.view.scatter('src_tags', src_tags, targets=self.target_ranks)
        self.view.scatter('src_distaxis_sizes', src_distaxis_sizes, targets=self.target_ranks)

        self.view.scatter('dest_targets', dest_targets, targets=other.target_ranks)
        self.view.scatter('dest_tags', dest_tags, targets=other.target_ranks)
        self.view.scatter('dest_distaxis_sizes', dest_distaxis_sizes, targets=other.target_ranks)

        # result ndarray
        result = ndarray(self.shape, self.distaxis, self.dtype, other.target_offsets, other.target_ranks)

        # send
        self.view.execute('%s_send_tasks = interengine.scatterArrayMPI_async(%s, src_targets[0], src_tags[0], src_distaxis_sizes[0], %d, target2rank)' \
            % (self.name, self.name, self.distaxis), targets=self.target_ranks)

        # receive
        self.view.execute("""\
            {0}_recv_tasks, {0}_recv_bufs = interengine.gatherArraysMPI_async({1}, dest_targets[0], dest_tags[0], dest_distaxis_sizes[0], {2}, target2rank)
            """.format(self.name, result.name, self.distaxis),\
            targets=result.target_ranks)

        # finish communication
        self.view.execute('''\
            if "{0}_send_tasks" in locals():
                MPI.Request.Waitall({0}_send_tasks)
                del {0}_send_tasks
            if "{0}_recv_tasks" in locals():
                MPI.Request.Waitall({0}_recv_tasks)
                interengine.finish_communication({1}, dest_distaxis_sizes[0], {2}, {0}_recv_bufs)
                del {0}_recv_tasks, {0}_recv_bufs
            '''.format(self.name, result.name, self.distaxis),
            targets=tuple(set(self.target_ranks + result.target_ranks)))

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
    # right-handed operators
    def __radd__(self, other):
        return dist_math.binary_rop(self, other, '+')
    def __rsub__(self, other):
        return dist_math.binary_rop(self, other, '-')
    def __rmul__(self, other):
        return dist_math.binary_rop(self, other, '*')
    def __rdiv__(self, other):
        return dist_math.binary_rop(self, other, '/')
    def __rfloordiv__(self, other):
        return dist_math.binary_rop(self, other, '//')
    def __rpow__(self, other):
        return dist_math.binary_rop(self, other, '**')
    # in-place operators
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

# import this module into dist_math and factories
my_module = sys.modules[__name__]
dist_math.ndarray = my_module
factories.ndarray = my_module
