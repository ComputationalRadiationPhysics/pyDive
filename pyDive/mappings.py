"""
Copyright 2014 Heiko Burau

This file is part of pyDive.

pyDive is free software: you can redistribute it and/or modify
it under the terms of of either the GNU General Public License or
the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
pyDive is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License and the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU General Public License
and the GNU Lesser General Public License along with pyDive.
If not, see <http://www.gnu.org/licenses/>.
"""

__doc__=\
"""If `numba <http://numba.pydata.org/>`_ is installed the particle shape functions will
be compiled which gives an appreciable speedup.
"""

import numpy as np
try:
    from numba import vectorize
except ImportError:
    vectorize = None

class NGP:
    """Nearest-Grid-Point
    """
    support = 1.0

    @staticmethod
    def __call__(x):
        if abs(x) < 0.5:
            return 1.0
        return 0.0

class CIC:
    """Cloud-in-Cell
    """
    support = 2.0

    @staticmethod
    def __call__(x):
        if abs(x) < 1.0:
            return 1.0 - abs(x)
        return 0.0

def __apply_MapOp(mesh, particles_pos, shape_function, mapOp):
    half_supp = 0.5 * shape_function.support
    num_coeffs_per_axis = int(np.ceil(shape_function.support))
    dim = len(np.shape(mesh))

    # ndarray which holds the non-zero particle assignment values of one particle
    coeffs = np.empty([num_coeffs_per_axis] * dim)

    if vectorize:
        # vectorize shape function to a compiled ufunc
        v_shape_function = vectorize(['f4(f4)', 'f8(f8)'])(shape_function.__call__)
    else:
        # numpy fallback
        v_shape_function = np.vectorize(shape_function.__call__)

    for idx, pos in enumerate(particles_pos):
        # lowest mesh indices that contributes to the mapping
        begin = np.ceil(pos - np.ones(dim) * half_supp).astype(int)

        # rearrange mapping area if it overlaps the border
        begin = np.maximum(begin, np.zeros(dim, dtype=int))
        begin = np.minimum(begin, np.shape(mesh) - np.ones(dim) * half_supp)

        # compute coefficients (particle assignment values)
        for coeff_idx in np.ndindex(np.shape(coeffs)):
            rel_vec = begin + coeff_idx - pos
            coeffs[coeff_idx] = np.prod(v_shape_function(rel_vec))

        # do the actual mapping
        window = [slice(begin[i], begin[i] + num_coeffs_per_axis) for i in range(dim)]
        mapOp(mesh[window], coeffs, idx)


def mesh2particles(mesh, particles_pos, shape_function=CIC):
    """
    Map mesh values to particles according to a particle shape function.

    :param array-like mesh: n-dimensional array.
        Dimension of *mesh* has to be greater or equal to the number of particle position components.
    :param particles_pos:
        'd'-dim tuples for 'N' particle positions. The positions can be float32 or float64 and
        must be within the shape of *mesh*.
    :type particles_pos: (N, d)
    :param shape_function:
        Callable object returning the particle assignment value for a given param 'x'.
        Has to provide a 'support' float attribute which defines the width of the non-zero area.
        Defaults to cloud-in-cell.
    :type shape_function: callable, optional
    :return: Mapped mesh values for each particle.
    :type return: ndarray(N, dtype='f')

    Notes:
        - The particle shape function is not evaluated outside the mesh.
    """
    # resulting ndarray
    particles = np.empty(len(particles_pos), dtype='f')

    def mapOp(sub_mesh, coeffs, particle_idx):
        particles[particle_idx] = np.add.reduce(sub_mesh * coeffs, axis=None)

    __apply_MapOp(mesh, particles_pos, shape_function, mapOp)

    return particles

def particles2mesh(mesh, particles, particles_pos, shape_function=CIC):
    """
    Map particle values to mesh according to a particle shape function.
    Particle values are added to the mesh.

    :param array-like mesh: n-dimensional array.
        Dimension of *mesh* has to be greater or equal to the number of particle position components.
    :param particles: particle data.
        len(*particles*) has to be the same as len(*particles_pos*)
    :type particles: array_like (1 dim)
    :param particles_pos:
        'd'-dim tuples for 'N' particle positions. The positions can be float32 or float64 and
        must be within the shape of *mesh*.
    :type particles_pos: (N, d)
    :param shape_function:
        Callable object returning the particle assignment value for a given param 'x'.
        Has to provide a 'support' float attribute which defines the width of the non-zero area.
        Defaults to cloud-in-cell.
    :type shape_function: callable, optional
    :return: *mesh*

    Notes:
        - The particle shape function is not evaluated outside the mesh.
    """

    def mapOp(sub_mesh, coeffs, particle_idx):
        sub_mesh += particles[particle_idx] * coeffs

    __apply_MapOp(mesh, particles_pos, shape_function, mapOp)

    return mesh
