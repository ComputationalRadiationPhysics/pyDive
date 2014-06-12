from ndarray import *
import IPParallelClient as com
from matplotlib import pyplot as plt
import numpy as np

#import aos_ndarray
from h5_ndarray import factories

d = factories.soa("/home/burau/test.h5", "/data/1000", 0)

#h = h5_ndarray("/home/burau/test.h5", "/data/1000/fields/Density_e", 1, np.s_[32, :, :])

