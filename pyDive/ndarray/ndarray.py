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

import numpy as np
import pyDive.distribution.distributor as distributor
from pyDive.distribution.interengine import MPI_copier

ndarray = distributor.distribute(np.ndarray, "ndarray", "np", interengine_copier=MPI_copier)

factories = distributor.generate_factories(ndarray, ("empty", "zeros", "ones"), np.float)
factories.update(distributor.generate_factories_like(ndarray, ("empty_like", "zeros_like", "ones_like")))

globals().update(factories)

def array(array_like, distaxis=0):
    np_array = np.array(array_like)
    result = empty(np_array.shape, np_array.dtype, distaxis)
    result[:] = np_array
    return result

factories.update({"array" : array})

ufunc_names = [key for key, value in np.__dict__.items() if isinstance(value, np.ufunc)]
ufuncs = distributor.generate_ufuncs(ufunc_names, "np")

globals().update(ufuncs)