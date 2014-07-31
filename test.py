import pyDive as pd
import numpy as np

h5fieldB = pd.picongpu.loadSteps([0], "/net/cns/projects/HPL/electrons/burau/KHI_dev/simOutput", "fields/FieldB", 0)
h5fieldB = h5fieldB.next()[1]
print h5fieldB
fieldB = h5fieldB[0,:,:]
print fieldB.shape

#field_energy = []
#par_energy = []

#for step, h5fields in pd.picongpu.loadAllSteps(\
    #"/net/cns/projects/HPL/electrons/burau/KHI_dev/simOutput", "fields", 0):

    #print step

    #energy = 0.0
    #energy += pd.mapReduce(lambda a: a['x']**2 + a['y']**2 + a['z']**2, np.add, h5fields["FieldE"])
    #energy += pd.mapReduce(lambda a: a['x']**2 + a['y']**2 + a['z']**2, np.add, h5fields["FieldB"])
    #field_energy.append(energy)

    #energy = pd.reduce(h5fields["EnergyDensity_e"], np.add)
    #par_energy.append(energy)

#from matplotlib import pyplot as plt
#plt.plot(field_energy + par_energy)
#plt.savefig("energy_tot.png")