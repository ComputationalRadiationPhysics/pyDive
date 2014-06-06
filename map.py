from ndarray import *
from h5_ndarray import *
from cloned_ndarray import *
import IPParallelClient as com
import numpy as np
from IPython.parallel import interactive

def __map_wrapper(f, array_names, **kwargs):
    arrays = [globals()[array_name] for array_name in array_names]
    f(*arrays, **kwargs)

def map(f, *arrays, **kwargs):
    view = com.getView()

    array_names = [repr(a) for a in arrays]

    view.targets = arrays[0].targets_in_use
    view.apply(interactive(__map_wrapper), f, array_names, **kwargs)
    view.targets = None