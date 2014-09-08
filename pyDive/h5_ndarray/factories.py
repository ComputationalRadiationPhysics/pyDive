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

from . import h5_ndarray
import h5py as h5
from .. import arrayOfStructs

def fromPath(h5_filename, datapath, distaxis=0, window=None):
    """Creates a :class:`pyDive.h5_ndarray` or structure of :ref:`pyDive.h5_ndarrays <pyDive.h5_ndarray>`
        from a hdf5-dataset respectively a hdf5-group.

        :param str h5_filename: hdf5-filename
        :param str datapath: path within the hdf5-file to a dataset or a group
        :param int distaxis: axis on which memory is distributed across the :term:`engines <engine>`
        :param window: This param let you specify a sub-part of the array as a virtual container.
            Example: window=np.s_[:,:,::2]
        :type window: list of slice objects (:ref:`numpy.s_`)
        :return: :obj:`pyDive.h5_ndarray` or structure of :obj:`pyDive.h5_ndarrays <pyDive.h5_ndarray>`\
            (:mod:`pyDive.arrayOfStructs`)

        Notes:
            - The dataset's attributes are stored in ``h5array.attrs``.
    """
    hFile = h5.File(h5_filename, 'r')
    datapath = datapath.rstrip("/")
    group_or_dataset = hFile[datapath]
    if type(group_or_dataset) is not h5._hl.group.Group:
        # dataset
        return h5_ndarray.h5_ndarray(h5_filename, datapath, distaxis, window)

    def create_tree(group, tree, dataset_path):
        for key, value in group.items():
            # group
            if type(value) is h5._hl.group.Group:
                tree[key] = {}
                create_tree(value, tree[key], dataset_path + "/" + key)
            # dataset
            else:
                tree[key] = h5_ndarray.h5_ndarray(h5_filename, dataset_path + "/" + key, distaxis, window)

    group = group_or_dataset
    structOfArrays = {}
    create_tree(group, structOfArrays, datapath)
    return arrayOfStructs.arrayOfStructs(structOfArrays)