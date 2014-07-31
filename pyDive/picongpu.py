"""This module holds convenient functions for those who use pyDive together with picongpu
(www.github.com/ComputationalRadiationPhysics/picongpu).
"""

import os
import os.path
import re
import h5_ndarray.factories

def loadSteps(steps, folder_path, data_path, distaxis, window=None):
    """Python generator object looping all hdf5-data found in *folder_path*
        from timesteps appearing in *steps*.

        This function doesn't read or write any data elements from hdf5 but returns dataset-handles
        covered by *pyDive.h5_ndarray* objects.

        All datasets inside *data_path* must have the same shape.

        :param ints steps: list of timesteps to loop
        :param str folder_path: Path of the folder containing the hdf5-files
        :param str data_path: Relative path starting from "/data/<timestep>/" within hdf5-file to the dataset or structure of datasets
        :param int distaxis: axis on which datasets are distributed over when once loaded into memory.
        :param window: This param let you specify a sub-part of the array as a virtual container.
            Example: window=np.s_[:,:,::2]
        :type window: list of slice objects (:ref:`numpy.s_`).
        :return: tuple of timestep and a :ref:`pyDive.h5_ndarray <pyDive.h5_ndarray.h5_ndarray.h5_ndarray>`
            or a structure of pyDive.h5_ndarrays. Ordering is done by timestep.
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

        yield timestep, h5_ndarray.factories.fromPath(full_filename, full_datapath, distaxis, window)

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

def loadAllSteps(folder_path, data_path, distaxis, window=None):
    """Python generator object looping hdf5-data of all timesteps found in *folder_path*.

        This function doesn't read or write any data elements from hdf5 but returns dataset-handles
        covered by *pyDive.h5_ndarray* objects.

        All datasets inside *data_path* must have the same shape.

        :param str folder_path: Path of the folder containing the hdf5-files
        :param str data_path: Relative path starting from "/data/<timestep>/" within hdf5-file to the dataset or structure of datasets
        :param int distaxis: axis on which datasets are distributed over when once loaded into memory.
        :param window: This param let you specify a sub-part of the array as a virtual container.
            Example: window=np.s_[:,:,::2]
        :type window: list of slice objects (:ref:`numpy.s_`).
        :return: tuple of timestep and a :ref:`pyDive.h5_ndarray <pyDive.h5_ndarray.h5_ndarray.h5_ndarray>`
            or a structure of pyDive.h5_ndarrays. Ordering is done by timestep.
    """
    steps = getSteps(folder_path)

    for timestep, data in loadSteps(steps, folder_path, data_path, distaxis, window):
        yield timestep, data