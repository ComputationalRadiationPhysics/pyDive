import os
import os.path
import re
import h5_ndarray.factories

def loadSteps(steps, folder_path, data_path, distaxis, window=None):
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
    assert os.path.exists(folder_path), "folder '%s' does not exist" % folder_path

    result = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.h5'): continue
        timestep = int(re.findall("\d+", filename)[-2])
        result.append(timestep)
    return result

def loadAllSteps(folder_path, data_path, distaxis, window=None):
    steps = getSteps(folder_path)

    for timestep, data in loadSteps(steps, folder_path, data_path, distaxis, window):
        yield timestep, data