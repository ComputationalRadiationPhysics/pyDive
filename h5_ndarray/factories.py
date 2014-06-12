from . import h5_ndarray
import h5py as h5

def soa(h5_filename, group_path, distaxis, window=None):
    hFile = h5.File(h5_filename, 'r')
    group = hFile[group_path]

    def create_tree(group, tree, dataset_path):
        for key, value in group.items():
            # group
            if type(value) is h5._hl.group.Group:
                tree[key] = {}
                create_tree(value, tree[key], dataset_path + "/" + key)
            # dataset
            else:
                tree[key] = h5_ndarray.h5_ndarray(h5_filename, dataset_path + "/" + key, distaxis, window)

    result = {}
    create_tree(group, result, group_path)
    return result