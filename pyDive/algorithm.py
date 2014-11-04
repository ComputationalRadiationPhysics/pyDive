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

import os
# check whether this code is executed on target or not
onTarget = os.environ.get("onTarget", 'False')
if onTarget == 'False':
    from ndarray import *
    from h5_ndarray.h5_ndarray import *
    from cloned_ndarray import *
    import IPParallelClient as com
    from IPython.parallel import interactive
    from h5_ndarray import h5caching
import numpy as np

def map(f, *arrays, **kwargs):
    """Calls *f* on :term:`engine` with local numpy-arrays related to *arrays*.
    Example: ::

        cluster_array = pyDive.ones(shape=[100], distaxis=0)

        cluster_array *= 2.0
        # equivalent to
        pyDive.map(lambda a: a *= 2.0, cluster_array) # a is the local numpy-array of *cluster_array*

    Or, as a decorator: ::

        @pyDive.map
        def twice(a):
            a *= 2.0

        twice(cluster_array)

    :param callable f: function to be called on :term:`engine`. Has to accept *numpy-arrays* and *kwargs*
    :param arrays: list of arrays including *pyDive.ndarrays*, *pyDive.h5_ndarrays* or *pyDive.cloned_ndarrays*
    :param kwargs: user-specified keyword arguments passed to *f*
    :raises AssertionError: if the *shapes* of *pyDive.ndarrays* and *pyDive.h5_ndarrays* do not match
    :raises AssertionError: if the *distaxis* attributes of *pyDive.ndarrays* and *pyDive.h5_ndarrays* do not match

    Notes:
        - If the hdf5 data exceeds the memory limit (see :obj:`pyDive.h5_ndarray.h5caching.fraction_of_av_mem_used`)\
            the data will be read block-wise so that a block fits into memory.
        - *map* chooses the list of *engines* from the **first** element of *arrays*. On these engines *f* is called.\
            If the first array is a *pyDive.h5_ndarray* all engines will be used.
        - *map* is not writing data back to a *pyDive.h5_ndarray* yet.
        - *map* does not equalize the element distribution of *pyDive.ndarrays* before execution.
    """
    if not arrays:
        # decorator mode
        def map_deco(*arrays, **kwargs):
            map(f, *arrays, **kwargs)
        return map_deco

    def map_wrapper(f, array_names, **kwargs):
        arrays = [globals()[array_name] for array_name in array_names]
        f(*arrays, **kwargs)

    view = com.getView()

    tmp_targets = view.targets # save current target list

    # loop all cache chunks
    for cached_arrays in h5caching.cache_arrays(*arrays):
        array_names = [repr(a) for a in cached_arrays]

        view.targets = cached_arrays[0].targets_in_use
        view.apply(interactive(map_wrapper), interactive(f), array_names, **kwargs)

    view.targets = tmp_targets # restore target list

def reduce(_array, op):
    """Perform a tree-like reduction over all axes of *_array*.

    :param _array: *pyDive.ndarray*, *pyDive.h5_ndarray* or *pyDive.cloned_ndarray* to be reduced
    :param numpy-ufunc op: reduce operation, e.g. *numpy.add*.

    If the hdf5 data exceeds the memory limit (see :obj:`pyDive.h5_ndarray.h5caching.fraction_of_av_mem_used`)\
    the data will be read block-wise so that a block fits into memory.
    """
    def reduce_wrapper(array_name, op_name):
        _array = globals()[array_name]
        op =  eval("np." + op_name)
        return algorithm.__tree_reduce(_array, axis=None, op=op) # reduction over all axes

    view = com.getView()

    tmp_targets = view.targets # save current target list

    result = None
    # loop all cache chunks
    for cached_arrays in h5caching.cache_arrays(_array):
        cached_array = cached_arrays[0] # there is just one array in the list
        array_name = repr(cached_array)

        view.targets = cached_array.targets_in_use

        targets_results = view.apply(interactive(reduce_wrapper), array_name, op.__name__)
        chunk_result = op.reduce(targets_results) # reduce over targets' results
        if result is None:
            result = chunk_result
        else:
            result = op(result, chunk_result)

    view.targets = tmp_targets # restore target list
    return result

def mapReduce(map_func, reduce_op, *arrays, **kwargs):
    """Calls *map_func* on :term:`engine` with local numpy-arrays related to *arrays*
    and reduces its result in a tree-like fashion over all axes.
    Example: ::

        cluster_array = pyDive.ones(shape=[100], distaxis=0)

        s = pyDive.mapReduce(lambda a: a**2, np.add, cluster_array) # a is the local numpy-array of *cluster_array*
        assert s == 100

    :param callable f: function to be called on :term:`engine`. Has to accept *numpy-arrays* and *kwargs*
    :param numpy-ufunc reduce_op: reduce operation, e.g. *numpy.add*.
    :param arrays: list of arrays including *pyDive.ndarrays*, *pyDive.h5_ndarrays* or *pyDive.cloned_ndarrays*
    :param kwargs: user-specified keyword arguments passed to *f*
    :raises AssertionError: if the *shapes* of *pyDive.ndarrays* and *pyDive.h5_ndarrays* do not match
    :raises AssertionError: if the *distaxis* attributes of *pyDive.ndarrays* and *pyDive.h5_ndarrays* do not match

    Notes:
        - If the hdf5 data exceeds the memory limit (see :obj:`pyDive.h5_ndarray.h5caching.fraction_of_av_mem_used`)\
            the data will be read block-wise so that a block fits into memory.
        - *mapReduce* chooses the list of *engines* from the **first** element of *arrays*. On these engines the mapReduce will be executed.\
            If the first array is a *pyDive.h5_ndarray* all engines will be used.
        - *mapReduce* is not writing data back to a *pyDive.h5_ndarray* yet.
        - *mapReduce* does not equalize the element distribution of *pyDive.ndarrays* before execution.
    """
    def mapReduce_wrapper(map_func, reduce_op_name, array_names, **kwargs):
        arrays = [globals()[array_name] for array_name in array_names]
        reduce_op =  eval("np." + reduce_op_name)
        return algorithm.__tree_reduce(map_func(*arrays, **kwargs), axis=None, op=reduce_op)

    view = com.getView()
    tmp_targets = view.targets # save current target list

    result = None
    # loop all cache chunks
    for cached_arrays in h5caching.cache_arrays(*arrays):
        array_names = [repr(a) for a in cached_arrays]

        view.targets = cached_arrays[0].targets_in_use
        targets_results = view.apply(interactive(mapReduce_wrapper),\
            interactive(map_func), reduce_op.__name__, array_names, **kwargs)

        chunk_result = reduce_op.reduce(targets_results) # reduce over targets' results
        if result is None:
            result = chunk_result
        else:
            result = reduce_op(result, chunk_result)

    view.targets = tmp_targets # restore target list

    return result

def __tree_reduce(array, axis=None, op=np.add):
    # reduce all axes
    if axis is None:
        result = array
        for axis in range(len(array.shape))[::-1]:
            result = __tree_reduce(result, axis=axis, op=op)
        return result

    assert 0 <= axis and axis < len(array.shape)
    result = np.rollaxis(array, axis)

    while True:
        l = result.shape[0]

        # if axis is small enough to neglect rounding errors do a faster numpy-reduce
        if l < 10000:
            return op.reduce(result, axis=0)

        if l % 2 == 0:
            result = op(result[0:l/2], result[l/2:])
        else:
            tmp = result[-1]
            result = op(result[0:l/2], result[l/2:-1])
            result[0] = op(result[0], tmp)
