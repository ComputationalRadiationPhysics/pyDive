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
try:
    from arrays.h5_ndarray import h5_ndarray
except ImportError:
    h5_ndarray = None
try:
    from arrays.ad_ndarray import ad_ndarray
except ImportError:
    ad_ndarray = None
import math

#: list of array types that store their elements on hard disk
hdd_arraytypes = (h5_ndarray, ad_ndarray)

def __bestStepSize(arrays, memory_limit):
    view = com.getView()

    # minimum amount of memory available and memory needed, both per engine
    get_mem_av_node = interactive(lambda: psutil.virtual_memory().available)
    mem_av = min(view.apply(get_mem_av_node)) / com.getPPN()
    mem_needed = sum(a.nbytes for a in arrays) / len(view.targets)

    # edge length of the whole array
    edge_length = arrays[0].shape[arrays[0].distaxis]
    # maximum edge length on one engine according to the available memory
    step_size = memory_limit * edge_length * mem_av / mem_needed

    if step_size >= edge_length:
        return edge_length

    # round 'step_size' down to nearest power of two
    return pow(2, int(math.log(step_size, 2)))

def fragment(*arrays, **kwargs):
    """Create fragments of *arrays* so that each fragment will fit into the combined
    main memory of all engines when calling ``load()``. The fragmentation is done by array slicing along the distributed axis.
    The edge size of the fragments is a power of two except for the last fragment.

    :param array: distributed arrays (e.g. pyDive.ndarray, pyDive.h5_ndarray, ...)
    :param kwargs: optional keyword arguments are: ``memory_limit`` and ``offset``.
    :param float memory_limit: fraction of the combined main memory of all engines reserved for fragmentation.
        Defaults to ``0.25``.
    :param bool offset: If ``True`` the returned tuple is extended by the fragments' offset (along the distributed axis).
        Defaults to ``False``.
    :raises AssertionError: If not all arrays have the same shape.
    :raises AssertionError: If not all arrays are distributed along the same axis.
    :return: generator object (list) of tuples. Each tuple consists of one fragment for each array in *arrays*.

    Note that *arrays* may contain an arbitrary number of distributed arrays of any type.
    While the fragments' size is solely calculated based on the memory consumption of
    arrays that store their elements on hard disk (see :obj:`hdd_arraytypes`),
    the fragmentation itself is applied on all arrays in the same way.

    Example: ::

        big_h5_array = pyDive.h5.open("monster.h5", "/")
        # big_h5_array.load() # crash

        for h5_array, offset in pyDive.fragment(big_h5_array, offset=True):
            a = h5_array.load() # no crash
            print "This fragment's offset is", offset, "on axis:", a.distaxis
    """
    # default keyword arguments
    kwargs_defaults = {"memory_limit" : 0.25, "offset" : False}
    kwargs_defaults.update(kwargs)
    kwargs = kwargs_defaults

    memory_limit = kwargs["memory_limit"]
    offset = kwargs["offset"]

    if not arrays: return

    assert all(a.distaxis == arrays[0].distaxis for a in arrays), \
        "all arrays must be distributed along the same axis"

    assert all(a.shape == arrays[0].shape for a in arrays), \
        "all arrays must have the same shape"

    # calculate the best suitable step size (-> fragment's edge size) according to the amount
    # of available memory on the engines
    hdd_arrays = [a for a in arrays if (hasattr(a, "arraytype") and a.arraytype in hdd_arraytypes) or type(a) in hdd_arraytypes]
    step = __bestStepSize(hdd_arrays, memory_limit)

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
