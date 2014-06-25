from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np
import arrayOfStructs

import h5_ndarray
from h5_ndarray import factories

d = factories.soa("/home/burau/test.h5", "/data/1000/fields/FieldE", 0)

#h = h5_ndarray.h5_ndarray.h5_ndarray("/home/burau/test.h5", "/data/1000/fields/Density_e", 0)

a = arrayOfStructs.arrayOfStructs(d)

b = a['x']

print type(b)