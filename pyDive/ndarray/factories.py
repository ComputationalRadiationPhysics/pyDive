"""This module holds high-level functions for instanciating :ref:`pyDive.ndarrays <pyDive.ndarray>`."""

import numpy as np
from .. import IPParallelClient as com
import helper
#import ndarray    # this import is done by the ndarray module itself due to circular dependencies

def hollow(shape, distaxis, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` without allocating a local
        *numpy-array*.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    return ndarray.ndarray(shape, distaxis, dtype, no_allocation=True)

def empty(shape, distaxis, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` without initializing elements.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    return ndarray.ndarray(shape, distaxis, dtype)

def zeros(shape, distaxis, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` filled with zeros.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = hollow(shape, distaxis, dtype)
    view = com.getView()
    view.execute("%s = np.zeros(%s.shape, %s.dtype)" % [result.name]*3, targets=result.targets_in_use)
    return result

def ones(shape, distaxis, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` filled with ones.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = hollow(shape, distaxis, dtype)
    view = com.getView()
    view.execute("%s = np.ones(%s.shape, %s.dtype)" % [result.name]*3, targets=result.targets_in_use)
    return result

def hollow_like(a):
    """Return a new :ref:`pyDive.ndarray` with the same shape, distribution and type as *a* without allocating
    a local *numpy-array*.
    """
    return ndarray.ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use, no_allocation=True)

def empty_like(a):
    """Return a new :ref:`pyDive.ndarray` with the same shape, distribution and type as *a* without initializing elements.
    """
    return ndarray.ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use)

def zeros_like(a):
    """Return a new :ref:`pyDive.ndarray` with the same shape, distribution and type as *a* filled with zeros.
    """
    result = hollow_like(a)
    view = com.getView()
    view.execute("%s = np.zeros(%s.shape, %s.dtype)" % [result.name]*3, targets=result.targets_in_use)
    return result

def ones_like(a):
    """Return a new :ref:`pyDive.ndarray` with the same shape, distribution and type as *a* filled with ones.
    """
    result = hollow_like(a)
    view = com.getView()
    view.execute("%s = np.ones(%s.shape, %s.dtype)" % [result.name]*3, targets=result.targets_in_use)
    return result

def array(array_like, distaxis):
    """Return a new :ref:`pyDive.ndarray` from an array-like object.

    :param array-like array_like: Any object exposing the array interface, e.g. numpy-array, python sequence, ...
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    """
    # numpy array
    if isinstance(array_like, np.ndarray):
        # result ndarray
        result = ndarray.ndarray(array_like.shape, distaxis, array_like.dtype, no_allocation=True)

        tmp = np.rollaxis(array_like, distaxis)
        sub_arrays = [tmp[begin:end] for begin, end in result.idx_ranges]
        # roll axis back
        sub_arrays = [np.rollaxis(ar, 0, distaxis+1) for ar in sub_arrays]

        view = com.getView()
        view.scatter('sub_array', sub_arrays, targets=result.targets_in_use)
        view.execute("%s = sub_array[0].copy()" % result.name, targets=result.targets_in_use)
        return result
    else:
        return array(np.array(array_like), distaxis)
