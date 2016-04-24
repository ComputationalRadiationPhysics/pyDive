# Copyright 2014-2016 Heiko Burau
#
# This file is part of pyDive.
#
# pyDive is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pyDive.  If not, see <http://www.gnu.org/licenses/>.


# ndarray
from .arrays import ndarray as ndarray_mod
globals().update(ndarray_mod.factories)
globals().update(ndarray_mod.ufuncs)
from .arrays.ndarray import ndarray  # noqa

# hdf5
try:
    from .arrays import h5_ndarray as h5  # noqa
except ImportError:
    pass

# adios
try:
    from .arrays import ad_ndarray as adios  # noqa
except ImportError:
    pass

# gpu
try:
    from .arrays import gpu_ndarray as gpu  # noqa
except ImportError:
    pass

# fragmentation
from .fragment import fragment  # noqa

# algorithm
from .algorithm import *  # noqa

# picongpu
from . import picongpu  # noqa

# init
from . import ipyParallelClient  # noqa
init = ipyParallelClient.init


# module doc
__doc__ = \
    """Make most used functions and modules directly accessable from pyDive."""

items = [item for item in globals().items() if not item[0].startswith("__")]
items.sort(key=lambda item: item[0])

fun_names = [":obj:`" + item[0] + "<" + item[1].__module__ + "." + item[0] + ">`"
             for item in items if hasattr(item[1], "__module__")]

import inspect  # noqa
module_names = [":mod:`" + item[0] + "<" + item[1].__name__ + ">`"
                for item in items if inspect.ismodule(item[1])]

__doc__ += "\n\n**Functions**:\n\n" + "\n\n".join(fun_names)\
    + "\n\n**Modules**:\n\n" + "\n\n".join(module_names)
