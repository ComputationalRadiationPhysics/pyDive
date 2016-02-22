# Copyright 2014-2016 Heiko Burau
#
# This file is part of pyDive.
#
# pyDive is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pyDive.  If not, see <http://www.gnu.org/licenses/>.


def getFirstSliceIdx(slice_obj, begin, end):
    if slice_obj.start > begin:
        if slice_obj.start >= end:
            return None
        return slice_obj.start
    i = (begin-1 - slice_obj.start) // slice_obj.step + 1
    idx = slice_obj.start + i * slice_obj.step
    if idx >= end or idx >= slice_obj.stop:
        return None
    return idx


def window_of_shape(shape, window):
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
        new_shape.append((clean_slice.stop-1 - clean_slice.start) // clean_slice.step + 1)
    return new_shape, clean_view


def window_of_view(view, window):
    result_view = []
    window = iter(window)
    for v in view:
        if type(v) is int:
            result_view.append(v)
            continue
        w = next(window)
        if type(w) is int:
            result_view.append(v.start + w * v.step)
            continue

        result_view.append(slice(v.start + w.start * v.step, v.start + w.stop * v.step, v.step * w.step))

    return result_view
