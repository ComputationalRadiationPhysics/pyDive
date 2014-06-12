from ndarray import *
from h5_ndarray import *
from cloned_ndarray import *
import IPParallelClient as com
import numpy as np
from IPython.parallel import interactive

fraction_of_av_mem_used = 0.5

def __map_wrapper(f, array_names, **kwargs):
    arrays = [globals()[array_name] for array_name in array_names]
    f(*arrays, **kwargs)

def __bestStepSize(h5_ndarrays):
    view = com.getView()

    # memory available and memory needed per engine
    get_mem_av = interactive(lambda: psutil.virtual_memory().available)
    mem_av = min(view.apply(get_mem_av))
    mem_needed = sum(a.nbytes for a in h5_ndarrays) / len(view.targets)

    edge_length = h5_ndarrays[0].shape[h5_ndarrays[0].distaxis]
    step_size = fraction_of_av_mem_used * edge_length * mem_av / mem_needed

    if step_size >= edge_length:
        return edge_length

    # round 'step_size' down to nearest power of two
    return pow(2, int(math.log(step_size, 2)))

def map(f, *arrays, **kwargs):
    view = com.getView()

    ndarrays = [a for a in arrays if isinstance(a, ndarray.ndarray)]
    h5_ndarrays = [a for a in arrays if isinstance(a, h5_ndarray.h5_ndarray)]
    both_ndarrays = ndarrays + h5_ndarrays

    if both_ndarrays:
        assert all(a.distaxis == both_ndarrays[0].distaxis for a in both_ndarrays), \
            "all ndarrays and h5_ndarrays must be distributed along the same axis"

        assert all(a.shape == both_ndarrays[0].shape for a in both_ndarrays), \
            "all ndarrays and h5_ndarrays must have the same shape"

        if h5_ndarrays:
            # calculate the best suitable step size (-> cache size) according to the amount
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
                # do nothing. Note that for h5_ndarrays the index operator returns a ndarray which is
                # read out in parallel from the hdf5 file.
                cache_arrays = [a[cache_window] if a in both_ndarrays else a     for a in arrays]

                # call this function again but now without h5_ndarrays
                map(f, *cache_arrays, **kwargs)
            return

    array_names = [repr(a) for a in arrays]

    tmp_targets = view.targets
    view.targets = arrays[0].targets_in_use
    view.apply(interactive(__map_wrapper), f, array_names, **kwargs)
    view.targets = tmp_targets