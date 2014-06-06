from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np
from h5_ndarray.h5_ndarray import *
import cloned_ndarray.factories
from cloned_ndarray import cloned_ndarray
import map

#h = h5_ndarray("/home/burau/test.h5", "/data/1000/fields/Density_e", 1)

#a = h[32, :, :]

c = cloned_ndarray.cloned_ndarray((8, 8), np.float, [0,1,2,3])

def assign(a, b):
    a = b

def hallo(a, factor):
    a[:] = ones_like(a) * factor

map.map(hallo, c, factor = 2.0)

#h[32:34, :16, :] = a

#a = a[:, 1::3, 5::3]

#plt.imshow(a.gather()[0,:,:])
print c.sum()
plt.imshow(c.sum())
plt.colorbar()
plt.savefig('data.png')

