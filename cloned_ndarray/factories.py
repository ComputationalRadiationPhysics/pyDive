import cloned_ndarray

def empty_targets_like(shape, dtype, a):
    return cloned_ndarray.cloned_ndarray(shape, dtype, a.targets_in_use)