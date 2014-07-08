from ndarray import *
from h5_ndarray.h5_ndarray import *
from cloned_ndarray import *
import IPParallelClient as com
import numpy as np
from IPython.parallel import interactive
from h5_ndarray import h5caching

def map(f, *arrays, **kwargs):
    def map_wrapper(f, array_names, **kwargs):
        arrays = [globals()[array_name] for array_name in array_names]
        f(*arrays, **kwargs)

    view = com.getView()

    tmp_targets = view.targets # save current target list

    # loop all cache chunks
    for cached_arrays in h5caching.cache_arrays(*arrays):
        array_names = [repr(a) for a in cached_arrays]

        view.targets = cached_arrays[0].targets_in_use
        view.apply(interactive(map_wrapper), f, array_names, **kwargs)

    view.targets = tmp_targets # restore target list

def reduce(_array, op):
    """
    Performs a reduction over all axes.

    Parameters
    ----------
    _array : h5_ndarray, ndarray, cloned_ndarray
    op : numpy ufunc (e.g. np.add)
        reduce function
    """
    def reduce_wrapper(array_name, op):
        _array = globals()[array_name]
        return op.reduce(_array, axis=None) # reduce over all axes

    view = com.getView()

    tmp_targets = view.targets # save current target list

    result = None
    # loop all cache chunks
    for cached_arrays in h5caching.cache_arrays(_array):
        cached_array = cached_arrays[0] # there is just one array in the list
        array_name = repr(cached_array)

        view.targets = cached_array.targets_in_use
        targets_results = view.apply(interactive(reduce_wrapper), array_name, op)
        chunk_result = op.reduce(targets_results) # reduce over targets' results
        if result is None:
            result = chunk_result
        else:
            result = op(result, chunk_result)

    view.targets = tmp_targets # restore target list
    return result

def mapReduce(map_func, reduce_op, *arrays, **kwargs):
    def mapReduce_wrapper(map_func, reduce_op, array_names, **kwargs):
        arrays = [globals()[array_name] for array_name in array_names]
        return reduce_op.reduce(map_func(*arrays, **kwargs), axis=None)

    view = com.getView()
    tmp_targets = view.targets # save current target list

    result = None
    # loop all cache chunks
    for cached_arrays in h5caching.cache_arrays(*arrays):
        array_names = [repr(a) for a in cached_arrays]

        view.targets = cached_arrays[0].targets_in_use
        targets_results = view.apply(interactive(mapReduce_wrapper),\
            map_func, reduce_op, array_names, **kwargs)

        chunk_result = reduce_op.reduce(targets_results) # reduce over targets' results
        if result is None:
            result = chunk_result
        else:
            result = reduce_op(result, chunk_result)

    view.targets = tmp_targets # restore target list
    return result