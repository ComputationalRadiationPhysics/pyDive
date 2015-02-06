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
"""This module holds convenient functions for those who use pyDive together with `picongpu \
<http://www.github.com/ComputationalRadiationPhysics/picongpu>`_.
"""

import os
import os.path
import re
import arrays.h5_ndarray as h5
import arrayOfStructs

def loadSteps(steps, folder_path, data_path, distaxis=0):
    """Python generator object looping all hdf5-data found in *folder_path*
        from timesteps appearing in *steps*.

        This generator doesn't read or write any data elements from hdf5 but returns dataset-handles
        covered by *pyDive.h5_ndarray* objects.

        All datasets within *data_path* must have the same shape.

        :param ints steps: list of timesteps to loop
        :param str folder_path: Path to the folder containing the hdf5-files
        :param str data_path: Relative path starting from "/data/<timestep>/" within hdf5-file to the dataset or group of datasets
        :param int distaxis: axis on which datasets are distributed over when once loaded into memory.
        :return: tuple of timestep and a :ref:`pyDive.h5_ndarray <pyDive.h5_ndarray.h5_ndarray.h5_ndarray>`
            or a structure of pyDive.h5_ndarrays (:mod:`pyDive.arrayOfStructs`). Ordering is done by timestep.

        Notes:
            - If the dataset has a '**sim_unit**' attribute its value is stored in ``h5array.unit``.
    """
    assert os.path.exists(folder_path), "folder '%s' does not exist" % folder_path

    timestep_and_filename = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.h5'): continue

        timestep = int(re.findall("\d+", filename)[-2])
        if not timestep in steps: continue
        timestep_and_filename.append((timestep, filename))

    # sort by timestep
    timestep_and_filename.sort(key=lambda item: item[0])

    for timestep, filename in timestep_and_filename:
        full_filename = os.path.join(folder_path, filename)
        full_datapath = os.path.join("/data", str(timestep), data_path)

        h5data = h5.open(full_filename, full_datapath, distaxis)

        # add 'sim_unit' as 'unit' attribute
        def add_sim_unit(array):
            if 'sim_unit' in array.attrs:
                setattr(array, "unit", array.attrs["sim_unit"])
            return array
        if type(h5data) is h5.h5_ndarray:
            h5data = add_sim_unit(h5data)
        else:
            h5data = arrayOfStructs.arrayOfStructs(\
                arrayOfStructs.makeTree_fromTree(h5data.structOfArrays, add_sim_unit))

        yield timestep, h5data

def getSteps(folder_path):
    """Returns a list of all timesteps in *folder_path*.
    """
    assert os.path.exists(folder_path), "folder '%s' does not exist" % folder_path

    result = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.h5'): continue
        timestep = int(re.findall("\d+", filename)[-2])
        result.append(timestep)
    return result

def loadAllSteps(folder_path, data_path, distaxis=0):
    """Python generator object looping hdf5-data of all timesteps found in *folder_path*.

        This generator doesn't read or write any data elements from hdf5 but returns dataset-handles
        covered by *pyDive.h5_ndarray* objects.

        All datasets within *data_path* must have the same shape.

        :param str folder_path: Path to the folder containing the hdf5-files
        :param str data_path: Relative path starting from "/data/<timestep>/" within hdf5-file to the dataset or group of datasets
        :param int distaxis: axis on which datasets are distributed over when once loaded into memory.
        :return: tuple of timestep and a :ref:`pyDive.h5_ndarray <pyDive.h5_ndarray.h5_ndarray.h5_ndarray>`
            or a structure of pyDive.h5_ndarrays (:mod:`pyDive.arrayOfStructs`). Ordering is done by timestep.

        Notes:
            - If the dataset has a '**sim_unit**' attribute its value is stored in ``h5array.unit``.
    """
    steps = getSteps(folder_path)

    for timestep, data in loadSteps(steps, folder_path, data_path, distaxis):
        yield timestep, data

def loadStep(step, folder_path, data_path, distaxis=0):
    """Load hdf5-data from a single timestep found in *folder_path*.

        All datasets within *data_path* must have the same shape.

        :param int step: timestep
        :param str folder_path: Path to the folder containing the hdf5-files
        :param str data_path: Relative path starting from "/data/<timestep>/" within hdf5-file to the dataset or group of datasets
        :param int distaxis: axis on which datasets are distributed over when once loaded into memory.
        :return: :ref:`pyDive.h5_ndarray <pyDive.h5_ndarray.h5_ndarray.h5_ndarray>`
            or a structure of pyDive.h5_ndarrays (:mod:`pyDive.arrayOfStructs`).

        Notes:
            - If the dataset has a '**sim_unit**' attribute its value is stored in ``h5array.unit``.
    """
    step, field = loadSteps([step], folder_path, data_path, distaxis).next()
    return field