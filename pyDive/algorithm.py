"""
Copyright 2014-2016 Heiko Burau

This file is part of pyDive.

pyDive is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyDive is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with pyDive.  If not, see <http://www.gnu.org/licenses/>.
"""
__doc__ = None

from . import ipyParallelClient as com
from ipyparallel import interactive
from .structured import VirtualArrayOfStructs, flat_values
from .distribution.generic_array import record

def map(f, *arrays, **kwargs):
    """Applies *f* on local arrays of *arrays*. It is very similar
    to python's' builtin ``map()`` except that the iteration is done over local arrays (in parallel)
    and not over single data values.

    :param callable f: function to be called on :term:`engine`.
    :param arrays: distributed arrays
    :param kwargs: keyword arguments are passed directly to *f*.
    :return: new distributed array if ``f`` returns non-``None`` on each engine else ``None``.
    :raises AssertionError: if the shapes of *arrays* do not match
    :raises AssertionError: if the decompositions of *arrays* do not match

    Example: ::

        celsius = pyDive.array([39.2, 36.5, 37.3, 37.8])

        fahrenheit = 9.0/5 * celsius + 32
        # equivalent to
        fahrenheit = pyDive.map(lambda x: 9.0/5 * x + 32, celsius) # `x` is the local array.

    Issues:
     - ``f`` must not return a *VirtualArrayOfStructs* yet.
    """

    with_decomposition = [a for a in arrays if hasattr(a, "decomposition")]
    decompositions = []
    for a in with_decomposition:
        decomp = a.decomposition
        if type(decomp) is dict:
            # `decomp` is a tree (dict of dicts), so flatten it first
            decompositions += list(flat_values(decomp))
        else:
            decompositions.append(decomp)

    assert all(decompositions[0] == d for d in decompositions),\
        "All arrays must have the same decomposition."

    def map_wrapper(f, array_names, **kwargs):
        arrays = tuple(map(globals().get, array_names))
        map_result = f(*arrays, **kwargs)
        globals()["map_result"] = map_result
        return (type(map_result), map_result.dtype) if map_result is not None else None

    # reference array.
    ref_array = arrays[0].firstArray if type(arrays[0]) == VirtualArrayOfStructs else arrays[0]

    view = com.getView()
    tmp_targets = view.targets # save current target list
    view.targets = ref_array.ranks()

    array_names = [repr(a) for a in arrays]
    local_results = view.apply(interactive(map_wrapper), interactive(f), array_names, **kwargs)

    result = None
    if all(r is not None for r in local_results):
        array_type = record[local_results[0][0]]
        dtype = local_results[0][1]
        result = array_type(ref_array.shape, dtype, ref_array.distaxes, ref_array.decomposition, True)
        view.execute("{} = map_result; del map_result".format(repr(result)))

    view.targets = tmp_targets # restore target list
    return result

def reduce(op, array, op_array=None):
    """Perform a reduction over all axes of *array*. It is done in two steps: first all local arrays are reduced
    by *op_array*, then the results are reduced further by *op*.

    :param op: binary reduce function.
    :param array: distributed array to be reduced.
    :param op_array: unary function which reduces the local array. If left to ``None`` it will be set
    to *op.reduce*. This is valid e.g. for all numpy operations (*np.add*, ...).

    Example: ::

        numbers = pyDive.array(range(10))

        assert pyDive.reduce(np.add, numbers) == 45
    """

    def reduce_wrapper(array_name, op):
        array = globals()[array_name]
        return op.reduce(array, axis=None) # reduction over all axes

    def reduce_wrapper_generic(array_name, op_array):
        array = globals()[array_name]
        return op_array(array, axis=None) # reduction over all axes

    view = com.getView()

    tmp_targets = view.targets # save current target list
    if type(array) == VirtualArrayOfStructs:
        view.targets = array.firstArray.ranks()
    else:
        view.targets = array.ranks()

    array_name = repr(array)

    if op_array is None:
        targets_results = view.apply(interactive(reduce_wrapper), array_name, op)
    else:
        targets_results = view.apply(interactive(reduce_wrapper_generic), array_name, op_array)

    import functools
    result = functools.reduce(op, targets_results) # reduce over targets' results

    view.targets = tmp_targets # restore target list
    return result
