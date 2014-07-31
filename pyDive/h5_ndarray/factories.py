from . import h5_ndarray
import h5py as h5
from .. import arrayOfStructs

def fromPath(h5_filename, datapath, distaxis, window=None):
    """
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