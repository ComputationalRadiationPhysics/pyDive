import pyDive as pd
import pyDive.picongpu
import numpy as np

print pyDive.picongpu.getSteps("/net/cns/projects/HPL/electrons/burau/KHI_dev/simOutput")

for step, h5fieldB in pyDive.picongpu.loadAllSteps(\
    "/net/cns/projects/HPL/electrons/burau/KHI_dev/simOutput", "fields/FieldB", 0):

    print step, pd.mapReduce(lambda a: a['x']**2 + a['y']**2 + a['z']**2, np.add, h5fieldB)
