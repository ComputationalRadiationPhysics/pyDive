import IPParallelClient as com
import numpy as np

def __replace_values_in_tree(tree, visitor):
    # traverse tree
    for key, value in tree.items():
        if type(value) is dict:
            __replace_values_in_tree(value, visitor)
        else:
            tree[key] = visitor(key, value)
    return tree

class aos_ndarray(object):
    def __init__(self, soa):
        assert all(a.shape == soa.values()[0].shape for a in soa.values()),
            "all arrays in structure-of-arrays ('soa') must have the same shape"