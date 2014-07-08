from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np
import arrayOfStructs
import algorithm
from h5_ndarray.h5_ndarray import h5_ndarray
from h5_ndarray import factories
from ndarray.ndarray import ndarray

s = factories.soa("/home/burau/test.h5", "/data/1000/fields/FieldE", 0, np.s_[0,:,:])
s = arrayOfStructs.arrayOfStructs(s)

print algorithm.mapReduce(lambda a: a['x']**2 + a['y']**2 + a['z']**2, np.add, s)
#print algorithm.reduce(s['x'], np.add)