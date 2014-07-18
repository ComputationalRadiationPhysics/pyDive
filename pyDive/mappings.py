from numpy import *
from numba import jit, vectorize

class NGP:
    support = 1.0

    @staticmethod
    def __call__(x):
        if abs(x) < 0.5:
            return 1.0
        return 0.0

class CIC:
    support = 2.0

    @staticmethod
    def __call__(x):
        if abs(x) < 1.0:
            return 1.0 - abs(x)
        return 0.0

def __apply_MapOp(mesh, particles_pos, shape_function, mapOp):
    half_supp = 0.5 * shape_function.support
    num_coeffs_per_axis = int(ceil(shape_function.support))
    dim = len(shape(mesh))

    # ndarray which holds the non-zero particle assignment values of one particle
    coeffs = empty([num_coeffs_per_axis] * dim)

    # vectorize shape function to a compiled ufunc
    v_shape_function = vectorize(['f4(f4)', 'f8(f8)'])(shape_function.__call__)

    for idx, pos in enumerate(particles_pos):
        # lowest mesh indices that contributes to the mapping
        begin = ceil(pos - ones(dim) * half_supp).astype(int)

        # rearrange mapping area if it overlaps the border
        begin = maximum(begin, zeros(dim, dtype=int))
        begin = minimum(begin, shape(mesh) - ones(dim) * half_supp)

        # compute coefficients (particle assignment values)
        for coeff_idx in ndindex(shape(coeffs)):
            rel_vec = begin + coeff_idx - pos
            coeffs[coeff_idx] = prod(v_shape_function(rel_vec))

        # do the actual mapping
        window = [slice(begin[i], begin[i] + num_coeffs_per_axis) for i in range(dim)]
        mapOp(mesh[window], coeffs, idx)


def mesh2particles(mesh, particles_pos, shape_function=CIC):
    """
    Maps mesh values to particles according to a particle shape function.

    Parameters
    ----------
    mesh : array_like
        Dimension of 'mesh' has to be greater or equal to shape(particle_pos)[1].
    particles_pos : (N, d)
        'd'-dim tuples for 'N' particle positions. The positions can be float32 or float64 and
        must be within the shape of 'mesh'.
    shape_function : callable, optional
        Callable object returning the particle assignment value for a given param 'x'.
        Has to provide a 'support' float attribute which defines the width of the non-zero area.
        Defaults to cloud-in-cell.

    Returns
    -------
    out : ndarray(N, dtype='f')
        Mapped mesh values for each particle.

    Notes
    -----
    The particle shape function is not evaluated outside the mesh.
    """
    # resulting ndarray
    particles = ndarray(len(particles_pos), dtype='f')

    def mapOp(sub_mesh, coeffs, particle_idx):
        particles[particle_idx] = sum(sub_mesh * coeffs)

    __apply_MapOp(mesh, particles_pos, shape_function, mapOp)

    return particles

def particles2mesh(mesh, particles, particles_pos, shape_function=CIC):
    """
    Maps particle values to mesh according to a particle shape function.
    Particle values are added to the mesh.

    Parameters
    ----------
    mesh : array_like
        Dimension of 'mesh' has to be greater or equal to shape(particle_pos)[1].
    particles: : array_like (1 dim)
        len(particles) has to be the same as len(particles_pos)
    particles_pos : (N, d)
        'd'-dim tuples for 'N' particle positions. The positions can be float32 or float64 and
        must be within the shape of 'mesh'.
    shape_function : callable, optional
        Callable object returning the particle assignment value for a given param 'x'.
        Has to provide a 'support' float attribute which defines the width of the non-zero area.
        Defaults to cloud-in-cell.

    Returns
    -------
    out : mesh

    Notes
    -----
    The particle shape function is not evaluated outside the mesh.
    """

    def mapOp(sub_mesh, coeffs, particle_idx):
        sub_mesh += particles[particle_idx] * coeffs

    __apply_MapOp(mesh, particles_pos, shape_function, mapOp)

    return mesh
