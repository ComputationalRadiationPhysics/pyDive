from h5_ndarray.factories import *
import algorithm
import numpy as np

s = fromPath("/home/burau/test.h5", "/data/1000/fields/FieldE/x", 0, np.s_[:,:,:])
#s = fromPath("/net/cns/projects/HPL/electrons/huebl/2014-07-02_khi-shortRad/simOutput/h5_data_1400.h5",\
#    "/data/1400/fields/FieldB", 0, np.s_[0,:,:])

print s.shape
a = s[:] # read out the whole array from hd5f into memory in parallel

#print algorithm.mapReduce(lambda a: a['x']**2 + a['y']**2 + a['z']**2, np.add, s)
#print algorithm.mapReduce(lambda a: a**2, np.add, s)
