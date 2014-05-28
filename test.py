from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np
from h5_ndarray.h5_ndarray import *

h = h5_ndarray("/home/burau/test.h5", "/data/1000/fields/Density_e", 1)

a = h[32:34, :512:8, :]

c = a + b

#h[32:34, :512:8, :] = 2.0 * a

a = a[:, 1::3, 5::3]

plt.imshow(a.gather()[0,:,:])
plt.colorbar()
plt.savefig('data.png')

