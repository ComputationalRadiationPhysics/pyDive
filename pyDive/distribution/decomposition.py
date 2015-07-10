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
import pyDive.IPParallelClient as com

class completeDC:

    def __init__(self, shape, distaxes, offsets=None, num_patches=None):
        """
        :param ints shape: shape for all distributed axes
        :param ints distaxes: distributed axes. Accepts a single integer too. Defaults to 'all' meaning each axis is distributed.
        :param offsets: For each distributed axis there is a (inner) list in the outer list.
            The inner list contains the offsets of the local array.
        :param num_patches: Number of patches in total. If ``None`` the number of available :term:`engines` is used.
            Ignored if *offsets* is not ``None``.
        :type offsets: list of lists
        """
        self.shape = shape
        self.distaxes = distaxes

        if offsets is None:
            if num_patches is None:
                view = com.getView()
                num_patches = len(self.view.targets)
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
        # to be save recalculate the number of patches
        num_patches_pa = [len(offsets_axis) for offsets_axis in offsets]
        num_patches = int(np.prod(num_patches_pa))
        self.num_patches = num_patches

    def __eq__(self, other):
        if self.shape != other.shape: return False
        if self.distaxes != other.distaxes: return False
        if self.num_patches != other.num_patches: return False
        if any(not np.array_equal(a, b) for a, b in zip(self.offsets, other.offsets)): return False
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