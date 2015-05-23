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
from collections import OrderedDict

def getFirstSliceIdx(slice_obj, begin, end):
    if slice_obj.start > begin:
        if slice_obj.start >= end: return None
        return slice_obj.start
    i = (begin-1 - slice_obj.start) / slice_obj.step + 1
    idx = slice_obj.start + i * slice_obj.step
    if idx >= end or idx >= slice_obj.stop: return None
    return idx

def view_of_shape(shape, window):
    new_shape = []
    clean_view = []
    for s, w in zip(shape, window):
        if type(w) is int:
            clean_view.append(w)
            continue
        # create a clean, wrapped slice object
        clean_slice = slice(*w.indices(s))
        clean_view.append(clean_slice)
        # new size of axis i
        new_shape.append((clean_slice.stop-1 - clean_slice.start) / clean_slice.step + 1)
    return new_shape, clean_view

def view_of_view(view, window):
    result_view = []
    window = iter(window)
    for v in view:
        if type(v) is int:
            result_view.append(v)
            continue
        w = window.next()
        if type(w) is int:
            result_view.append(v.start + w * v.step)
            continue

        result_view.append(slice(v.start + w.start * v.step, v.start + w.stop * v.step, v.step * w.step))

    return result_view

# create local slice objects for each engine
def createLocalSlices(slices, distaxis, target_offsets, shape):
    local_slices = [list(slices) for i in range(len(target_offsets))]
    distaxis_slice = slices[distaxis]
    for i in range(len(target_offsets)):
        begin = target_offsets[i]
        end = target_offsets[i+1] if i+1 < len(target_offsets) else shape[distaxis]

        local_slices[i][distaxis] = slice(distaxis_slice.start + distaxis_slice.step * begin,\
                                          distaxis_slice.start + distaxis_slice.step * end,\
                                          distaxis_slice.step)

    return local_slices

def common_decomposition(axesA, offsetsA, axesB, offsetsB, shape):
    axes = list(OrderedDict.fromkeys(axesA + axesB)) # remove double axes while preserving order
    offsets = []
    ids = []
    for axis in axes:
        if not (axis in axesA and axis in axesB):
            offsets.append(offsetsA[axesA.index(axis)] if axis in axesA else offsetsB[axesB.index(axis)])
            ids.append([])
            continue

        offsetsA_sa = offsetsA[axesA.index(axis)]
        offsetsB_sa = offsetsB[axesB.index(axis)]

        offsets_sa = [] # sa = single axis
        ids_sa = []
        A_idx = 0 # partition index
        B_idx = 0
        begin = 0
        # loop the common decomposition of A and B.
        # the current partition is given by [begin, end)
        while not begin == shape[axis]:
            end_A = offsetsA_sa[A_idx+1] if A_idx < len(offsetsA_sa)-1 else shape[axis]
            end_B = offsetsB_sa[B_idx+1] if B_idx < len(offsetsB_sa)-1 else shape[axis]
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
    return axes, offsets, ids