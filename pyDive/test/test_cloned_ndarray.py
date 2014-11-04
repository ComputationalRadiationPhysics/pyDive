import pyDive
import numpy as np
import pytest
import random
from pyDive import IPParallelClient as com

sizes = ((1,), (5,),
        (5, 29), (64, 1), (64, 64),
        (1, 1, 1), (12, 37, 50))

def test_cloned_ndarray(init_pyDive):
    view = com.getView()
    for size in sizes:
        ref_array = np.arange(np.prod(size))

        test_array = pyDive.cloned.empty(size, dtype=np.int)
        test_array[:] = ref_array

        assert np.array_equal(ref_array * len(view), test_array.sum())