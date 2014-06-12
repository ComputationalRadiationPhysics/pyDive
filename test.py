from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np
from h5_ndarray.h5_ndarray import *
from cloned_ndarray import cloned_ndarray, factories
import map

h = h5_ndarray("/home/burau/test.h5", "/data/1000/fields/Density_e", 1, np.s_[32, :, :])

a = h[:,:,:]

c = factories.zeros_targets_like((1, 256, 512), a.dtype, a)

def add(a, b):
    a[:b.shape[0], :b.shape[1], :b.shape[2]] += b

#def hallo(a, factor):
    #a[:] = ones_like(a) * factor

map.map(add, c, h)

#h[32:34, :16, :] = a

#a = a[:, 1::3, 5::3]

#plt.imshow(a.gather()[0,:,:])
#print c.sum()
plt.imshow(c.sum()[0,:,:])
plt.colorbar()
plt.savefig('data.png')

