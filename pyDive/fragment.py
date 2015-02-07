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

import IPParallelClient as com
from IPython.parallel import interactive
from arrays.h5_ndarray import h5_ndarray
import math

hdd_arraytypes = (h5_ndarray,)

def __bestStepSize(arrays, fraction):
    view = com.getView()

    # minimum amount of memory available and memory needed, both per engine
    get_mem_av_node = interactive(lambda: psutil.virtual_memory().available)
    mem_av = min(view.apply(get_mem_av_node)) / com.getPPN()
    mem_needed = sum(a.nbytes for a in arrays) / len(view.targets)

    # edge length of the whole array
    edge_length = arrays[0].shape[arrays[0].distaxis]
    # maximum edge length on one engine according to the available memory
    step_size = fraction * edge_length * mem_av / mem_needed

    if step_size >= edge_length:
        return edge_length

    # round 'step_size' down to nearest power of two
    return pow(2, int(math.log(step_size, 2)))

def fragment(*arrays, **kwargs):
    # default keyword arguments
    kwargs_defaults = {"fraction" : 0.25, "offset" : False}
    kwargs_defaults.update(kwargs)
    kwargs = kwargs_defaults

    fraction = kwargs["fraction"]
    offset = kwargs["offset"]

    if not arrays: return

    assert all(a.distaxis == arrays[0].distaxis for a in arrays), \
        "all arrays must be distributed along the same axis"

    assert all(a.shape == arrays[0].shape for a in arrays), \
        "all arrays must have the same shape"

    # calculate the best suitable step size (-> fragment's edge size) according to the amount
    # of available memory on the engines
    hdd_arrays = [a for a in arrays if a.arraytype in hdd_arraytypes or type(a) in hdd_arraytypes]
    step = __bestStepSize(hdd_arrays, fraction)

    shape = arrays[0].shape
    distaxis = arrays[0].distaxis
    # list of slices representing the fragment's shape
    fragment_window = [slice(None)] * len(shape)

    for begin in range(0, shape[distaxis], step):
        end = min(begin + step, shape[distaxis])
        fragment_window[distaxis] = slice(begin, end)

        result = [a[fragment_window] for a in arrays]
        if offset:
            offset_vector = [0] * len(shape)
            offset_vector[distaxis] = begin
            result += [offset_vector]
        if len(result) == 1:
            yield result[0]
        else:
            yield result

    return
