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

def getFirstSubIdx(slice_obj, begin, end):
    if slice_obj.start > begin:
        if slice_obj.start >= end: return None
        return slice_obj.start
    i = (begin-1 - slice_obj.start) / slice_obj.step + 1
    idx = slice_obj.start + i * slice_obj.step
    if idx >= end or idx >= slice_obj.stop: return None
    return idx

def subWindow_of_shape(shape, window):
    new_shape = []
    clean_slices = list(window)
    for i in range(len(window)):
        if type(window[i]) is int:
            clean_slices[i] = slice(window[i], window[i]+1, 1)
            continue
        # create a clean, wrapped slice object
        wrapped_ids = window[i].indices(shape[i])
        clean_slices[i] = slice(*wrapped_ids)
        # new size of axis i
        new_shape.append((clean_slices[i].stop-1 - clean_slices[i].start) / clean_slices[i].step + 1)
    return new_shape, clean_slices

def createLocalSlices(slices, distaxis, idx_ranges):
    # create local slice objects for each engine

    local_slices = [list(slices) for i in range(len(idx_ranges))]
    distaxis_slice = slices[distaxis]
    for i in range(len(idx_ranges)):
        begin, end = idx_ranges[i]

        local_slices[i][distaxis] = slice(distaxis_slice.start + distaxis_slice.step * begin,\
                                          distaxis_slice.start + distaxis_slice.step * end,\
                                          distaxis_slice.step)

    return local_slices