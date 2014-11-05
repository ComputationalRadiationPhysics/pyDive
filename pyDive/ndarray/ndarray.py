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
    def __init__(self, shape, distaxis=0, dtype=np.float, idx_ranges=None, targets_in_use=None, no_allocation=False):
        """Creates an :class:`pyDive.ndarray.ndarray.ndarray` instance. This is a low-level method for instanciating
        an array. Arrays should  be constructed using 'empty', 'zeros'
        or 'array' (see :mod:`pyDive.ndarray.factories`).

        :param shape: size of the array on each axis
        :type shape: tuple of ints
        :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
        :param numpy-dtype dtype: datatype of a single data value
        :param idx_ranges: list of (begin, end) pairs indicating the index range the corresponding
            :term:`engine` (see :ref:`targets_in_use`) is associated with
        :type idx_ranges: tuple/list of (int, int)
        :param targets_in_use: list of :term:`engine`-ids that share this array.
        :type targets_in_use: tuple/list of ints
        :param bool no_allocation: if ``True`` no actual memory, i.e. *numpy-array*, will be
            allocated on :term:`engine`. Useful when you want to assign an existing numpy array manually.
        :raises ValueError: if just *idx_ranges* is given and *targets_in_use* not or vice versa

        If *idx_ranges* and *targets_in_use* are both ``None`` they will be auto-generated
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

        assert distaxis >= 0 and distaxis < len(self.shape)

        if idx_ranges is None and targets_in_use is None:
            # number of available targets (engines)
            num_targets_av = len(self.view.targets)

            # shape of the mpi-local ndarray
            localshape = np.array(self.shape)
            localshape[distaxis] = (self.shape[distaxis] - 1) / num_targets_av + 1
            tmp = localshape[distaxis]

            # number of occupied targets by this ndarray instance
            num_targets = (self.shape[distaxis] - 1) / localshape[distaxis] + 1

            #: list of pairs on which each pair stores the range of indices [begin, end) for the distributed axis on each engine
            #: this is the decomposition of the distributed axis
            self.idx_ranges = [(r * tmp, (r+1) * tmp) for r in range(0, num_targets-1)]
            self.idx_ranges += [((num_targets-1) * tmp, self.shape[distaxis])]
            #: list of indices of the occupied engines
            self.targets_in_use = list(range(num_targets))
        elif idx_ranges is not None and targets_in_use is not None:
            self.idx_ranges = list(idx_ranges)
            self.targets_in_use = list(targets_in_use)
        else:
            raise ValueError("either args 'idx_ranges' and 'targets_in_use' have to be given both or not given both.")

        # generate a unique variable name used on target representing this instance
        global ndarray_id
        #: Unique variable name of the local *numpy-array* on *engine*.
        #: Unless you do manual stuff on the *engines* there is no need for dealing with this attribute.
        self.name = 'dist_ndarray' + str(ndarray_id)
        ndarray_id += 1

        if no_allocation:
            self.view.push({self.name : None}, targets=self.targets_in_use)
        else:
            # instanciate an empty ndarray object of the appropriate shape on each target in use
            localshapes = [self.shape[:] for i in range(len(self.targets_in_use))]
            for i in range(len(self.targets_in_use)):
                localshapes[i][distaxis] = self.idx_ranges[i][1] - self.idx_ranges[i][0]

            self.view.scatter('localshape', localshapes, targets=self.targets_in_use)
            self.view.push({'dtype' : dtype}, targets=self.targets_in_use)
            self.view.execute('%s = np.empty(localshape[0], dtype=dtype)' % self.name, targets=self.targets_in_use)

    def __del__(self):
        self.view.execute('del %s' % self.name, targets=self.targets_in_use)

    def __getitem__(self, args):
        if not isinstance(args, list) and not isinstance(args, tuple):
            args = (args,)

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(args), len(self.shape))

        # if args is a list of indices then return a single data value
        if all(type(arg) is int for arg in args):
            dist_idx = args[self.distaxis]
            for i in range(len(self.idx_ranges)):
                begin, end = self.idx_ranges[i]
                if dist_idx >= end: continue
                local_idx = list(args)
                local_idx[self.distaxis] = dist_idx - begin
                return self.view.pull("%s%s" % (self.name, repr(local_idx)), targets=self.targets_in_use[i])

        # shape of the new sliced ndarray
        new_shape, clean_slices = helper.subWindow_of_shape(self.shape, args)

        # clean slice object in the direction of the distributed axis
        distaxis_slice = clean_slices[self.distaxis]

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
            self.view.push({'args' : args}, targets=new_targets_in_use[i])
        self.view.execute('%s = %s[args]' % (result.name, self.name), targets=new_targets_in_use)

        return result

    def __setitem__(self, key, value):
        # if args is [:] then assign value to the entire ndarray
        if key == slice(None):
            if isinstance(value, np.ndarray):
                value = factories.array(value, self.distaxis)
            other = value.dist_like(self)
            self.view.execute("%s[:] = %s" % (self.name, other.name), targets=self.targets_in_use)
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
                self.view.push({'value' : value}, targets=self.targets_in_use[i])
                self.view.execute("%s%s = value" % (self.name, repr(local_idx)), targets=self.targets_in_use[i])
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
        local_arrays = self.view.pull(self.name, targets=self.targets_in_use)
        return np.concatenate(local_arrays, axis=self.distaxis)

    def copy(self):
        """Returns a hard copy of this array.
        """
        result = ndarray(self.shape, self.distaxis, self.dtype, self.idx_ranges, self.targets_in_use, True)
        self.view.execute("%s = %s.copy()" % (result.name, self.name), targets=self.targets_in_use)
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
        assert self.shape == other.shape
        assert self.distaxis == other.distaxis # todo: add this feature

        # if self is already distributed like *other* do nothing
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
        self.view.scatter('src_targets', src_targets, targets=self.targets_in_use)
        self.view.scatter('src_tags', src_tags, targets=self.targets_in_use)
        self.view.scatter('src_distaxis_sizes', src_distaxis_sizes, targets=self.targets_in_use)

        self.view.scatter('dest_targets', dest_targets, targets=other.targets_in_use)
        self.view.scatter('dest_tags', dest_tags, targets=other.targets_in_use)
        self.view.scatter('dest_distaxis_sizes', dest_distaxis_sizes, targets=other.targets_in_use)

        # result ndarray
        result = ndarray(self.shape, self.distaxis, self.dtype, other.idx_ranges, other.targets_in_use)

        # send
        self.view.execute('%s_send_tasks = interengine.scatterArrayMPI_async(%s, src_targets[0], src_tags[0], src_distaxis_sizes[0], %d, target2rank)' \
            % (self.name, self.name, self.distaxis), targets=self.targets_in_use)

        # receive
        self.view.execute("""\
            {0}_recv_tasks, {0}_recv_bufs = interengine.gatherArraysMPI_async({1}, dest_targets[0], dest_tags[0], dest_distaxis_sizes[0], {2}, target2rank)
            """.format(self.name, result.name, self.distaxis),\
            targets=result.targets_in_use)

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
            targets=list(set(self.targets_in_use + result.targets_in_use)))

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
