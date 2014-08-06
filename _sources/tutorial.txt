Tutorials
=========

Example 1: Total field energy
-----------------------------

Let's suppose we have a hdf5-file containing a 3D array representing an electric field and we want to compute
its total energy. This means squaring and summing or in pyDive's words: ::

  import pyDive
  import numpy as np
  pyDive.init()

  h5field = pdDive.h5.fromPath("field.h5", "FieldE", distaxis=0)
  field = h5field[:] # read out the entire array into cluster memory in parallel

  energy_field = field["x"]**2 + field["y"]**2 + field["z"]**2
  
  total_energy = pyDive.reduce(energy_field, np.add)

Now what happens if *h5field* is too large to be stored in the cluster's memory? The line ``field = h5field[:]`` will crash.
In this case we want to load the hdf5 data piece by piece. The functions in :mod:`pyDive.algorithm` help us doing so: ::

  import pyDive
  import numpy as np
  pyDive.init()

  h5field = pdDive.h5.fromPath("field.h5", "FieldE", distaxis=0)

  def square_field(npfield):
    return npfield["x"]**2 + npfield["y"]**2 + npfield["z"]**2

  total_energy = pyDive.mapReduce(square_field, np.add, h5field)

*square_field* is called on each :term:`engine` where *npfield* is a structure (:mod:`pyDive.arrayOfStructs`) of numpy-arrays representing a sub part of the big *h5field*.
:func:`pyDive.algorithm.mapReduce` can be called with an arbitrary number of arrays including
:obj:`pyDive.ndarrays`, :obj:`pyDive.h5_ndarrays` and :obj:`pyDive.cloned_ndarrays`. If there are :obj:`pyDive.h5_ndarrays` it will
check whether they fit into the cluster memory as a whole and loads them piece by piece if not.

Now let's say our dataset is really big and we just want to get a first estimate of the total energy: ::

  ...
  total_energy = pyDive.mapReduce(square_field, np.add, h5field[::10, ::10, ::10]) * 10.0**3

This is valid if *h5field[::10, ::10, ::10]* fits into the cluster's memory. Note that slicing on a :obj:`pyDive.h5_ndarray` always
means reading or writing from hdf5 to respectively from memory. So if it doesn't fit we have to apply the slicing somewhere else in fact
at the time when instanciating *h5field*: ::

  import pyDive
  import numpy as np
  pyDive.init()

  h5field = pdDive.h5.fromPath("field.h5", "FieldE", distaxis=0, window=np.s_[::10, ::10, ::10])

  def square_field(npfield):
    return npfield["x"]**2 + npfield["y"]**2 + npfield["z"]**2

  total_energy = pyDive.mapReduce(square_field, np.add, h5field) * 10.0**3

This way the hdf5 data is sliced without involving file i/o.

If you use `picongpu <https://github.com/ComputationalRadiationPhysics/picongpu>`_
here is an example of how to get the total field energy for each timestep: ::

  import pyDive
  import numpy as np
  pyDive.init()

  def square_field(npfield):
    return npfield["x"]**2 + npfield["y"]**2 + npfield["z"]**2

  for step, h5field in pyDive.picongpu.loadAllSteps("/.../simOutput", "fields/FieldE", distaxis=0):
    total_energy = pyDive.mapReduce(square_field, np.add, h5field)

    print step, total_energy

Example 2: Particle density field (picongpu)
--------------------------------------------

Given a huge list of particles in a hdf5-file we want to create a 3D density field out of it. We assume that the particle
positions are distributed randomly. This means although each engine is loading a separate part of all particles it needs to 
write to the entire density field. Therefore the density field must have a whole representation on each participating engine.
This is the job of :obj:`pyDive.cloned_ndarray`. ::

  import pyDive
  import numpy as np
  pyDive.init()

  shape = [256, 256, 256]
  density = pyDive.cloned.zeros(shape)

  filename = "/.../simOutput/h5_1000.h5"
  globalCellIdx = pyDive.h5.fromPath(filename, "/data/1000/particles/e/globalCellIdx", distaxis=0)
  position = pyDive.h5.fromPath(filename, "/data/1000/particles/e/position", distaxis=0)
  weighting = pyDive.h5.fromPath(filename, "/data/1000/particles/e/weighting", distaxis=0)

  def particles2density(globalCellIdx, pos, weighting, density):
    total_pos_x = globalCellIdx["x"].astype(pos.dtype["x"]) + pos["x"]
    total_pos_y = globalCellIdx["y"].astype(pos.dtype["y"]) + pos["y"]
    total_pos_z = globalCellIdx["z"].astype(pos.dtype["z"]) + pos["z"]

    # convert total_pos_x, total_pos_y and total_pos_z to an (N, 3) shaped array
    total_pos = np.hstack((total_pos_x[:,np.newaxis],
                           total_pos_y[:,np.newaxis],
                           total_pos_z[:,np.newaxis]))
    
    import pyDive.mappings
    pyDive.mappings.particles2mesh(density, weighting, total_pos, pyDive.mappings.CIC)  

  pyDive.map(particles2density, globalCellIdx, position, weighting, density)

  final_density = density.sum() # add up all local copies

Here, as in the example above, *particles2density* is a function executed on the :term:`engines <engine>` by :func:`pyDive.algorithm.map`.
All of its arguments are numpy-arrays or structures (:mod:`pyDive.arrayOfStructs`) of numpy-arrays.







