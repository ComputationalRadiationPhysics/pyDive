from . import h5_ndarray
import h5py as h5
import arrayOfStructs

def fromPath(h5_filename, path, distaxis, window=None):
    hFile = h5.File(h5_filename, 'r')
    path = path.rstrip("/")
    group_or_dataset = hFile[path]
    if type(group_or_dataset) is not h5._hl.group.Group:
        # dataset
        return h5_ndarray.h5_ndarray(h5_filename, path, distaxis, window)

    def create_tree(group, tree, dataset_path):
        for key, value in group.items():
            # group
            if type(value) is h5._hl.group.Group:
                tree[key] = {}
                create_tree(value, tree[key], dataset_path + "/" + key)
            # dataset
            else:
                tree[key] = h5_ndarray.h5_ndarray(h5_filename, dataset_path + "/" + key, distaxis, window)

    structOfArrays = {}
    create_tree(group, structOfArrays, group_path)
    return arrayOfStructs.arrayOfStructs(structOfArrays)