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
import itertools
import pyDive.IPParallelClient as com
import helper

class completeDC:

    def __init__(self, shape, distaxes, offsets=None, ranks=None, slices=None):
        """
        :param ints shape: shape for all distributed axes
        :param ints distaxes: distributed axes. Accepts a single integer too. Defaults to 'all' meaning each axis is distributed.
        :param offsets: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains the offsets for each local array.
        :param ranks: linear list of :term:`engine` ranks holding the local arrays.
        :param slices: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains slices for each local array. Slices are optional.
        :type offsets: list of lists
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
                num_patches = len(self.view.targets)
            else:
                num_patches = len(ranks)target_
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

        self.slices = slices

        num_ranks_pa = [len(offsets_sa) for offsets_sa in self.offsets]
        self.pitch = [int(np.prod(num_ranks_pa[i+1:])) for i in range(len(self.distaxes))]

    def __getitem__(self, args):
        """Performs a slicing operation on the decomposition.

        :return: New, sliced decomposition.
        """
        # shape of the new sliced ndarray
        new_shape, clean_view = helper.view_of_shape(self.shape, args)

        # determine properties of the new, sliced ndarray
        # keep these in mind when reading the following for loop
        local_slices_aa = [] # aa = all (distributed) axes
        new_offsets = []
        new_distaxes = []
        new_rank_ids_aa = []

        for distaxis, distaxis_idx, target_offsets \
            in zip(self.distaxes, range(len(self.distaxes)), self.offsets):

            # slice object in the direction of the distributed axis
            distaxis_slice = clean_view[distaxis]

            # if it turns out that distaxis_slice is actually an int, this axis has to vanish.
            # That means the local slice object has to be an int too.
            if type(distaxis_slice) is int:
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
            new_offsets.append(new_target_offsets_sa)
            new_rank_ids_aa.append(new_rank_ids_sa)

            # shift distaxis by the number of vanished axes in the left of it
            new_distaxis = distaxis - sum(1 for arg in args[:distaxis] if type(arg) is int)
            new_distaxes.append(new_distaxis)

        # create list of targets which participate slicing, this is new_ranks
        new_ranks = []
        indices = [xrange(len(offsets_pa)) for offsets_pa in self.offsets]
        for rank_idx_vector in itertools.product(*indices):
            rank_idx = sum(i * p for i, p in itertools.izip(rank_idx_vector, self.pitch))
            new_ranks.append(self.ranks[rank_idx])

        new_decomposition = completeDC(new_shape, new_distaxes, new_offsets, new_ranks, local_slices_aa)
        return new_decomposition

    def patches(self, nd_idx=False, offsets=False, next_offsets=False, position=False, slices=False):
        """Generator looping all patches returning a tuple of enabled properties for each patch.

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
            properties.append(itertools.product(*indices))
        if offsets:
            properties.append(itertools.product(*self.offsets))
        if next_offsets:
            next_offsets = [itertools.chain(itertools.islice(offsets_pa, start=1), (self.shape[distaxis],) ),\
                for offsets_pa, distaxis in zip(self.offsets, self.distaxes)]
            properties.append(itertools.product(*next_offsets))
        if position:
            position = [self.offsets[self.distaxes.index(axis)] if axis in self.distaxes else [0] for axis in range(len(self.shape))]
            properties.append(itertools.product(*position))
        if slices:
            slices = [self.slices[self.distaxes.index(axis)] if axis in self.distaxes else slice(None) for axis in range(len(self.shape))]
            properties.append(itertools.product(*slices))

        if len(properties) > 1:
            return itertools.izip(*properties)
        else:
            return properties[0]

    def __eq__(self, other):
        if self.shape != other.shape: return False
        if self.distaxes != other.distaxes: return False
        if self.num_patches != other.num_patches: return False
        if any(not np.array_equal(a, b) for a, b in zip(self.offsets, other.offsets)): return False
        if self.ranks != other.ranks: return False
        return True

    def __ne__(self, other):
        return not (self == other)

# ============================ end of completeDC ==================================================

def common_decomposition(dcA, dcB):
    assert dcA.shape == dcB.shape,\
        "Shapes of decompositions do not match: " + str(dcA.shape) + " <-> " + str(dcB.shape)

    axes = list(OrderedDict.fromkeys(dcA.distaxes + dcB.distaxes)) # remove double axes while preserving order
    offsets = []
    ids = []
    for axis in axes:
        if not (axis in dcA.distaxes and axis in dcB.distaxes):
            offsets.append(dcA.offsets[dcA.distaxes.index(axis)] if axis in dcA.distaxes else dcB.offsets[dcB.distaxes.index(axis)])
            ids.append([])
            continue

        offsetsA_sa = dcA.offsets[dcA.distaxes.index(axis)]
        offsetsB_sa = dcB.offsets[dcB.distaxes.index(axis)]

        offsets_sa = [] # sa = single axis
        ids_sa = []
        A_idx = 0 # partition index
        B_idx = 0
        begin = 0
        # loop the common decomposition of A and B.
        # the current partition is given by [begin, end)
        while not begin == dcA.shape[axis]:
            end_A = offsetsA_sa[A_idx+1] if A_idx < len(offsetsA_sa)-1 else dcA.shape[axis]
            end_B = offsetsB_sa[B_idx+1] if B_idx < len(offsetsB_sa)-1 else dcA.shape[axis]
            end = min(end_A, end_B)

            offsets_sa.append(begin)
            ids_sa.append((A_idx, B_idx))

            # go to next common partition
            if end == end_A:
                A_idx += 1
            if end == end_B:
                B_idx += 1

            begin = end

        offsets.append(np.array(offsets_sa))
        ids.append(np.array(ids_sa))

    return completeDC(dcA.shape, axes, offsets), ids