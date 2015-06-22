"""
Copyright 2015 Heiko Burau

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

import h5py as h5
import pyDive.distribution.helper as helper
import numpy as np

class h5_ndarray(object):

    def __init__(self, filename, dataset_path, shape=None, window=None, offset=None):
        self.filename = filename
        self.dataset_path = dataset_path

        fileHandle = h5.File(filename, "r")
        dataset = fileHandle[dataset_path]
        #self.attrs = dataset.attrs
        #: datatype of a single data value
        self.dtype = dataset.dtype
        if shape is None:
            shape = dataset.shape
        self.shape = tuple(shape)
        fileHandle.close()

        if window is None:
            window = [slice(0, s, 1) for s in shape]
        self.window = tuple(window)
        if offset is None:
            offset = (0,) * len(shape)
        self.offset = tuple(offset)

        #: total bytes consumed by the elements of the array.
        self.nbytes = self.dtype.itemsize * np.prod(self.shape)

    def load(self):
        window = list(self.window)
        for i in range(len(window)):
            if type(window[i]) is int:
                window[i] += self.offset[i]
            else:
                window[i] = slice(window[i].start + self.offset[i], window[i].stop + self.offset[i], window[i].step)

        fileHandle = h5.File(self.filename, "r")
        dataset = fileHandle[self.dataset_path]
        result = dataset[tuple(window)]
        fileHandle.close()
        return result

    def __getitem__(self, args):
        if args == slice(None):
            args = (slice(None),) * len(self.shape)

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = [args]

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(args), len(self.shape))

        assert not all(type(arg) is int for arg in args),\
            "single data access is not supported"

        result_shape, clean_view = helper.view_of_shape(self.shape, args)

        # Applying 'clean_view' after 'self.window', results in 'result_window'
        result_window = helper.view_of_view(self.window, clean_view)

        return h5_ndarray(self.filename, self.dataset_path, result_shape, result_window, self.offset)
