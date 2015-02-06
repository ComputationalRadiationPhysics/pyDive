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

from ..ndarray import *
from h5_ndarray import *
from .. import IPParallelClient as com
from IPython.parallel import interactive

#: fraction of the available memory per engine used for caching hdf5 files.
fraction_of_av_mem_used = 0.25

def __bestStepSize(h5_ndarrays):
    view = com.getView()

    # minimum amount of memory available and memory needed, both per engine
    get_mem_av_node = interactive(lambda: psutil.virtual_memory().available)
    mem_av = min(view.apply(get_mem_av_node)) / com.getPPN()
    mem_needed = sum(a.nbytes for a in h5_ndarrays) / len(view.targets)

    # edge length of the whole h5_ndarray
    edge_length = h5_ndarrays[0].shape[h5_ndarrays[0].distaxis]
    # maximum edge length on one engine according to the available memory
    step_size = fraction_of_av_mem_used * edge_length * mem_av / mem_needed

    if step_size >= edge_length:
        return edge_length

    # round 'step_size' down to nearest power of two
    return pow(2, int(math.log(step_size, 2)))

def cache_arrays(*arrays):
    ndarrays = [a for a in arrays if a.arraytype is ndarray.ndarray]
    h5_ndarrays = [a for a in arrays if a.arraytype is h5_ndarray]
    both_ndarrays = ndarrays + h5_ndarrays

    if both_ndarrays:
        assert all(a.distaxis == both_ndarrays[0].distaxis for a in both_ndarrays), \
            "all ndarrays and h5_ndarrays must be distributed along the same axis"

        assert all(a.shape == both_ndarrays[0].shape for a in both_ndarrays), \
            "all ndarrays and h5_ndarrays must have the same shape"

        if h5_ndarrays:
            # calculate the best suitable step size (-> cache's edge size) according to the amount
            # of available memory on the engines
            step = __bestStepSize(h5_ndarrays)

            h5_shape = h5_ndarrays[0].shape
            distaxis = h5_ndarrays[0].distaxis
            # list of slices representing the cache's shape
            cache_window = [slice(None)] * len(h5_shape)

            for begin in range(0, h5_shape[distaxis], step):
                end = min(begin + step, h5_shape[distaxis])
                cache_window[distaxis] = slice(begin, end)

                # for h5_ndarrays and ndarrays do caching and for other types of arrays (cloned_ndarray)
                # do nothing. Note that for h5_ndarrays the []-operator returns a ndarray which is
                # read out from the hdf5 file in parallel.
                yield [a[cache_window] if a in both_ndarrays else a     for a in arrays]
            return

    yield arrays
