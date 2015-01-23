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
"""This module holds high-level functions for instanciating :ref:`pyDive.cloned_ndarrays <pyDive.cloned_ndarrays>`."""

import pyDive.cloned_ndarray.cloned_ndarray as cloned_ndarray
from .. import IPParallelClient as com
import numpy as np

def hollow_engines_like(shape, dtype, a):
    """Return a new :obj:`pyDive.cloned_ndarray` utilizing the same engines *a* does
    without allocating a local *numpy-array*.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    :param a: :ref:`pyDive.ndarray`
    """
    return cloned_ndarray.cloned_ndarray(shape, dtype, a.target_ranks, True)

def empty_engines_like(shape, dtype, a):
    """Return a new :obj:`pyDive.cloned_ndarray` utilizing the same engines *a* does
    without initializing elements.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    :param a: :ref:`pyDive.ndarray`
    """
    return cloned_ndarray.cloned_ndarray(shape, dtype, a.target_ranks)

def zeros_engines_like(shape, dtype, a):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing the same engines *a* does
    filled with zeros.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    :param a: :ref:`pyDive.ndarray`
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, a.target_ranks, True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.target_ranks)
    view.execute('%s = np.zeros(myshape, dtype)' % repr(result), targets=result.target_ranks)
    return result

def hollow(shape, dtype=np.float):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing all engines without allocating a local
        *numpy-array*.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, target_ranks='all', no_allocation=True)
    return result

def empty(shape, dtype=np.float):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing all engines without initializing elements.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, target_ranks='all')
    return result

def zeros(shape, dtype=np.float):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing all engines filled with zeros.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, target_ranks='all', no_allocation=True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.target_ranks)
    view.execute('%s = np.zeros(myshape, dtype)' % repr(result), targets=result.target_ranks)
    return result

def ones(shape, dtype=np.float):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing all engines filled with ones.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, target_ranks='all', no_allocation=True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.target_ranks)
    view.execute('%s = np.ones(myshape, dtype)' % repr(result), targets=result.target_ranks)
    return result