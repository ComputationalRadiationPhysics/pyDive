import numpy as np
from .. import IPParallelClient as com
import helper
#import ndarray    # this import is done by the ndarray module itself due to circular dependencies

def hollow(shape, distaxis, dtype=np.float):
    return ndarray.ndarray(shape, distaxis, dtype, no_allocation=True)

def empty(shape, distaxis, dtype=np.float):
    return ndarray.ndarray(shape, distaxis, dtype)

def hollow_like(a):
    return ndarray.ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use, no_allocation=True)

def empty_like(a):
    return ndarray.ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use)

def array(array_like, distaxis):
    # numpy array
    if isinstance(array_like, np.ndarray):
        # result ndarray
        result = ndarray.ndarray(array_like.shape, distaxis, array_like.dtype, no_allocation=True)

        tmp = np.rollaxis(array_like, distaxis)
        sub_arrays = [tmp[begin:end] for begin, end in result.idx_ranges]
        # roll axis back
        sub_arrays = [np.rollaxis(ar, 0, distaxis+1) for ar in sub_arrays]

        view = com.getView()
        view.scatter('sub_array', sub_arrays, targets=result.targets_in_use)
        view.execute("%s = sub_array[0].copy()" % result.name, targets=result.targets_in_use)

        return result