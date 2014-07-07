from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np
import arrayOfStructs
import algorithm
from h5_ndarray.h5_ndarray import h5_ndarray
from h5_ndarray import factories
from ndarray.ndarray import ndarray

h = h5_ndarray("/home/burau/test.h5", "/data/1000/fields/FieldE/x", 0, np.s_[0,:,:])

a = ndarray(h.shape, 0)

def t(a,b):
    a = b

algorithm.map(t, a, h)

print algorithm.mapReduce(lambda a: a**2, np.add, h)