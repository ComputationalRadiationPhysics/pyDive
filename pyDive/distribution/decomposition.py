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

import numpy as np
from collections import OrderedDict
from itertools import product, izip
import pyDive.IPParallelClient as com
import helper

class completeDC:

    def __init__(self, shape, distaxes, offsets=None, ranks=None, slices=None):
        """
        :param ints shape: shape for all distributed axes
        :param ints distaxes: distributed axes. Accepts a single integer too.
        :param offsets: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains the offsets for each local array.
        :type offsets: list of lists
        :param ranks: linear list of :term:`engine` ranks holding the local arrays.
        :param slices: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains slices for each local array. Slices are optional.
        :type slices: list of lists
        """
        self.shape = shape
        if type(distaxes) not in (list, tuple):
            distaxes = (distaxes,)
        elif type(distaxes) is not tuple:
            distaxes = tuple(distaxes)
        self.distaxes = distaxes

        if offsets is None:
            if ranks is None:
                view = com.getView()
                num_patches = len(view.targets)
            else:
                num_patches = len(ranks)
            # create hypothetical patch with best surface-to-volume ratio
            patch_volume = np.prod(self.shape) / float(num_patches)
            patch_edge_length = pow(patch_volume, 1.0/len(self.distaxes))

            def factorize(n):
                if n == 1: yield 1; return
                for f in range(2,n//2+1) + [n]:
                    while n%f == 0:
                        n //= f
                        yield f
            prime_factors = list(factorize(num_patches))[::-1] # get prime factors of number of engines in descending order
            sorted_distaxes = sorted(self.distaxes, key=lambda axis: self.shape[axis]) # sort distributed axes in ascending order
            # calculate number of available targets (engines) per distributed axis
            # This value should be close to array_edge_length / patch_edge_length
            num_targets_av = [1] * len(self.shape)

            for distaxis in sorted_distaxes[:-1]:
                num_partitions = self.shape[distaxis] / patch_edge_length
                while float(num_targets_av[distaxis]) < num_partitions and prime_factors:
                    num_targets_av[distaxis] *= prime_factors.pop()
            # the largest axis gets the remaining (largest) prime_factors
            if prime_factors:
                num_targets_av[sorted_distaxes[-1]] *= np.prod(prime_factors)

            # calculate offsets
            localshape = np.array(self.shape)
            for distaxis in self.distaxes:
                localshape[distaxis] = (self.shape[distaxis] - 1) / num_targets_av[distaxis] + 1

            # number of occupied targets for each distributed axis by this ndarray instance
            num_targets = [(self.shape[distaxis] - 1) / localshape[distaxis] + 1 for distaxis in self.distaxes]

            # calculate offsets
            offsets = [np.arange(num_targets[i]) * localshape[self.distaxes[i]] for i in range(len(self.distaxes))]

        self.offsets = offsets

        # to be safe, recalculate the number of patches
        num_patches_pa = [len(offsets_axis) for offsets_axis in offsets]
        num_patches = int(np.prod(num_patches_pa))
        if ranks is None:
            ranks = tuple(range(num_patches))
        else:
            ranks = tuple(ranks[:num_patches])
        self.ranks = ranks

        if not slices:
            slices = [(slice(None),)] * len(self.shape)
        self.slices = slices

        num_parts_aa = [len(offsets_sa) for offsets_sa in self.offsets]
        self.pitch = [int(np.prod(num_parts_aa[i+1:])) for i in range(len(self.distaxes))]

    def __getitem__(self, args):
        """Performs a slicing operation on the decomposition.

        :return: New, sliced decomposition.
        """
        # shape of the new sliced ndarray
        new_shape, clean_view = helper.view_of_shape(self.shape, args)

        # determine properties of the new, sliced ndarray
        # keep these in mind when reading the following for loop
        new_slices = [(arg,) for arg in args]
        new_offsets = []
        new_distaxes = []
        new_rank_ids_aa = [] # aa = all (distributed) axes

        for distaxis, offsets_sa in zip(self.distaxes, self.offsets):

            # slice object in the direction of the distributed axis
            distaxis_slice = clean_view[distaxis]

            # if it turns out that distaxis_slice is actually an int, this axis has to vanish.
            # That means the local slice object has to be an int too.
            if type(distaxis_slice) is int:
                dist_idx = distaxis_slice
                rank_idx_component = np.searchsorted(offsets_sa, dist_idx, side="right") - 1
                local_idx = dist_idx - offsets_sa[rank_idx_component]
                new_slices[distaxis] = (local_idx,)
                new_rank_ids_aa.append((rank_idx_component,))
                continue

            # determine properties of the new, sliced ndarray
            new_slices_sa = [] # sa = single axis
            new_offsets_sa = []
            new_rank_ids_sa = []
            total_ids = 0

            first_rank_idx = np.searchsorted(offsets_sa, distaxis_slice.start, side="right") - 1
            last_rank_idx = np.searchsorted(offsets_sa, distaxis_slice.stop, side="right")

            for i in range(first_rank_idx, last_rank_idx):

                # index range of current target
                begin = offsets_sa[i]
                end = offsets_sa[i+1] if i < len(offsets_sa)-1 else self.shape[distaxis]
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
                new_slices_sa.append(slice(firstSliceIdx - begin, lastSliceIdx+1 - begin, distaxis_slice.step))
                # number of indices remaining on the current target after slicing
                num_ids = (lastSliceIdx - firstSliceIdx) / distaxis_slice.step + 1
                # new offset for current target
                new_offsets_sa.append(total_ids)
                total_ids += num_ids
                # target rank index
                new_rank_ids_sa.append(i)

            new_rank_ids_sa = np.array(new_rank_ids_sa)

            new_slices[distaxis] = new_slices_sa
            new_offsets.append(np.array(new_offsets_sa))
            new_rank_ids_aa.append(new_rank_ids_sa)

            # shift distaxis by the number of vanished axes in the left of it
            new_distaxis = distaxis - sum(1 for arg in args[:distaxis] if type(arg) is int)
            new_distaxes.append(new_distaxis)

        # create list of targets which participate slicing, this is new_ranks
        new_ranks = []
        for rank_idx_vector in product(*new_rank_ids_aa):
            rank_idx = sum(i * p for i, p in izip(rank_idx_vector, self.pitch))
            new_ranks.append(self.ranks[rank_idx])

        return completeDC(new_shape, new_distaxes, new_offsets, new_ranks, new_slices)

    def __str__(self):
        return "shape: {}\ndistaxes: {}\noffsets: {}\nranks: {}\nslices: {}".format(\
            self.shape, self.distaxes, self.offsets, self.ranks, self.slices)

    def patches(self, nd_idx=False, offsets=False, next_offsets=False, position=False, slices=False):
        """Iterator looping all patches returning a tuple of enabled properties for each patch.

        :param nd_idx: tuple of partition indices (on each distributed axis)
        :param offsets: tuple of partition offsets (on each distributed axis)
        :param next_offsets: tuple of offsets of the next partition (on each distributed axis).
            For the last partition the next offset is set to the edge's length.
        :param position: tuple of patch offsets on each axis.
            For a non-distributed axis the offset is ``0``.
        :param slices: tuple of local slices on each axis.
        """
        properties = []
        if nd_idx:
            indices = [xrange(len(offsets_pa)) for offsets_pa in self.offsets]
            properties.append(product(*indices))
        if offsets:
            properties.append(product(*self.offsets))
        if next_offsets:
            next_offsets = [tuple(offsets_pa[1:]) + (self.shape[distaxis],) \
                for offsets_pa, distaxis in zip(self.offsets, self.distaxes)]
            properties.append(product(*next_offsets))
        if position:
            position = [self.offsets[self.distaxes.index(axis)] if axis in self.distaxes else [0] for axis in range(len(self.shape))]
            properties.append(product(*position))
        if slices:
            properties.append(product(*self.slices))

        if len(properties) > 1:
            return izip(*properties)
        else:
            return properties[0]

    def __eq__(self, other):
        if self.shape != other.shape: return False
        if self.distaxes != other.distaxes: return False
        if any(not np.array_equal(a, b) for a, b in zip(self.offsets, other.offsets)): return False
        if self.ranks != other.ranks: return False
        return True

    def __ne__(self, other):
        return not (self == other)

# ============================ end of completeDC ==================================================

def common_axes(dcA, dcB):
    # remove double axes
    return sorted(list(set(dcA.distaxes + dcB.distaxes)))

def common_decomposition(dcA, dcB):
    """Compute the common (total) decomposition of two individual decompositions

    :param dcA: instance of ``completeDC``
    :param dcB: instance of ``completeDC``
    :return: (common decomposition, partition indices for each distributed axis, offsets
        for each distributed axis)
    """
    assert dcA.shape == dcB.shape,\
        "Shapes of decompositions do not match: " + str(dcA.shape) + " <-> " + str(dcB.shape)

    axes = common_axes(dcA, dcB)
    new_offsets = []
    nd_idx_AB = []
    offsets_AB = []
    for axis in axes:
        if not (axis in dcA.distaxes and axis in dcB.distaxes):
            dc = dcA if axis in dcA.distaxes else dcB

            offsets_sa = dc.offsets[dc.distaxes.index(axis)]
            new_offsets.append(offsets_sa)
            # partition indices of A and B
            ids = range(len(offsets_sa))
            nd_idx_AB.append((ids,[None]*len(ids)) if id(dc) == id(dcA) else ([None]*len(ids),ids))
            # offsets of A and B
            offsets_AB.append((offsets_sa, np.zeros(len(offsets_sa),dtype=int)) if id(dc) == id(dcA) else (np.zeros(len(offsets_sa),dtype=int),offsets_sa) )
            continue

        offsetsA_sa = dcA.offsets[dcA.distaxes.index(axis)]
        offsetsB_sa = dcB.offsets[dcB.distaxes.index(axis)]

        offsets_sa = [] # sa = single axis
        offsetsA_com = []
        offsetsB_com = []
        A_ids = []
        B_ids = []
        A_idx = 0 # partition index
        B_idx = 0
        begin = 0
        # loop the common decomposition of A and B along ``axis``.
        # the current partition is given by [begin, end)
        while not begin == dcA.shape[axis]:
            end_A = offsetsA_sa[A_idx+1] if A_idx < len(offsetsA_sa)-1 else dcA.shape[axis]
            end_B = offsetsB_sa[B_idx+1] if B_idx < len(offsetsB_sa)-1 else dcA.shape[axis]
            end = min(end_A, end_B)

            offsets_sa.append(begin)
            A_ids.append(A_idx)
            B_ids.append(B_idx)
            offsetsA_com.append(offsetsA_sa[A_idx])
            offsetsB_com.append(offsetsB_sa[B_idx])

            # go to next common partition
            if end == end_A:
                A_idx += 1
            if end == end_B:
                B_idx += 1

            begin = end

        new_offsets.append(np.array(offsets_sa))

        nd_idx_AB.append((A_ids, B_ids))
        offsets_AB.append((np.array(offsetsA_com), np.array(offsetsB_com)))

    return completeDC(dcA.shape, axes, new_offsets), nd_idx_AB, offsets_AB

def common_patches(dcA, dcB, nd_idx=False, offsets=False, next_offsets=False, ranks_AB = False, offsets_AB=False):
    """Iterator looping all common patches of two decompositions returning a tuple
        of enabled properties for each common patch.

    :param dcA: instance of ``completeDC``
    :param dcB: instance of ``completeDC``
    :param nd_idx: tuple of partition indices (on each distributed axis)
    :param offsets: tuple of partition offsets (on each distributed axis)
    :param next_offsets: tuple of offsets of the next partition (on each distributed axis).
        For the last partition the next offset is set to the edge's length.
    :param ranks_AB: pair of engine-ranks for ``dcA`` and ``dcB``.
    :param offsets_AB: pair of ndim-offsets (n = # distributed axes) for ``dcA`` and ``dcB``.
    """

    dc, nd_idx_AB, offsets_AB = common_decomposition(dcA, dcB)

    def get_ranks_AB(dcA, dcB, nd_idx_AB):
        nd_ids_A = zip(*nd_idx_AB)[0]
        nd_ids_B = zip(*nd_idx_AB)[1]

        for nd_idx_A, nd_idx_B in izip(product(*nd_ids_A), product(*nd_ids_B)):
            nd_idx_A = filter(lambda i: i is not None, nd_idx_A)
            nd_idx_B = filter(lambda i: i is not None, nd_idx_B)

            idx_A = sum(i * p for i, p in izip(nd_idx_A, dcA.pitch))
            idx_B = sum(i * p for i, p in izip(nd_idx_B, dcB.pitch))

            yield dcA.ranks[idx_A], dcB.ranks[idx_B]

    def get_offsets_AB(offsets_AB):
        offsets_A = zip(*offsets_AB)[0]
        offsets_B = zip(*offsets_AB)[1]

        return izip(product(*offsets_A), product(*offsets_B))

    properties = []
    if ranks_AB:
        properties.append(get_ranks_AB(dcA, dcB, nd_idx_AB))
    if offsets_AB:
        properties.append(get_offsets_AB(offsets_AB))

    return izip(dc.patches(nd_idx, offsets, next_offsets), *properties)