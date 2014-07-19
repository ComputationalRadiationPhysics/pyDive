from .. import cloned_ndarray
from .. import IPParallelClient as com

def empty_targets_like(shape, dtype, a):
    return cloned_ndarray.cloned_ndarray(shape, dtype, a.targets_in_use)

def zeros_targets_like(shape, dtype, a):
    result = cloned_ndarray.cloned_ndarray(shape, dtype, a.targets_in_use, True)
    view = com.getView()
    view.push({'myshape' : shape, 'dtype' : dtype}, targets=result.targets_in_use)
    view.execute('%s = np.zeros(myshape, dtype)' % repr(result), targets=result.targets_in_use)
    return result