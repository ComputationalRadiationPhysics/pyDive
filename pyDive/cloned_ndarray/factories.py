"""This module holds high-level functions for instanciating :ref:`pyDive.cloned_ndarrays <pyDive.cloned_ndarrays>`."""

import pyDive.cloned_ndarray.cloned_ndarray as cloned_ndarray
from .. import IPParallelClient as com
import numpy as np

def empty_engines_like(shape, dtype, a):
    """Return a new :obj:`pyDive.cloned_ndarray` utilizing the same engines *a* does
    without initializing elements.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    :param a: :ref:`pyDive.ndarray`
    """
    return cloned_ndarray.cloned_ndarray(shape, dtype, a.targets_in_use)

def zeros_engines_like(shape, dtype, a):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing the same engines *a* does
    filled with zeros.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    :param a: :ref:`pyDive.ndarray`
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, a.targets_in_use, True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.targets_in_use)
    view.execute('%s = np.zeros(myshape, dtype)' % repr(result), targets=result.targets_in_use)
    return result

def zeros(shape, dtype=np.float):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing all engines filled with zeros.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, targets_in_use='all', no_allocation=True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.targets_in_use)
    view.execute('%s = np.zeros(myshape, dtype)' % repr(result), targets=result.targets_in_use)
    return result

def ones(shape, dtype=np.float):
    """Return a new :ref:`pyDive.cloned_ndarray` utilizing all engines filled with ones.

    :param ints shape: shape of the array
    :param numpy-dtype dtype: datatype of a single data value
    """
    result = cloned_ndarray.cloned_ndarray(shape, dtype, targets_in_use='all', no_allocation=True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.targets_in_use)
    view.execute('%s = np.ones(myshape, dtype)' % repr(result), targets=result.targets_in_use)
    return result