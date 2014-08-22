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

__doc__=\
"""This module holds high-level functions for instanciating :ref:`pyDive.ndarrays <pyDive.ndarray>`."""

import numpy as np
from .. import IPParallelClient as com
import helper
#import ndarray    # this import is done by the ndarray module itself due to circular dependencies

def hollow(shape, distaxis=0, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` without allocating a local
        *numpy-array*.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    return ndarray.ndarray(shape, distaxis, dtype, no_allocation=True)

def empty(shape, distaxis=0, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` without initializing elements.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    return ndarray.ndarray(shape, distaxis, dtype)

def zeros(shape, distaxis=0, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` filled with zeros.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = empty(shape, distaxis, dtype)
    view = com.getView()
    view.execute("{0} = np.zeros({0}.shape, {0}.dtype)".format(result.name), targets=result.targets_in_use)
    return result

def ones(shape, distaxis=0, dtype=np.float):
    """Return a new :ref:`pyDive.ndarray` distributed across all :term:`engines <engine>` filled with ones.

    :param ints shape: shape of the array
    :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = empty(shape, distaxis, dtype)
    view = com.getView()
    view.execute("{0} = np.ones({0}.shape, {0}.dtype)".format(result.name), targets=result.targets_in_use)
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
    result = empty_like(a)
    view = com.getView()
    view.execute("{0} = np.zeros({0}.shape, {0}.dtype)".format(result.name), targets=result.targets_in_use)
    return result

def ones_like(a):
    """Return a new :ref:`pyDive.ndarray` with the same shape, distribution and type as *a* filled with ones.
    """
    result = empty_like(a)
    view = com.getView()
    view.execute("{0} = np.zeros({0}.shape, {0}.dtype)".format(result.name), targets=result.targets_in_use)
    return result

def array(array_like, distaxis=0):
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
