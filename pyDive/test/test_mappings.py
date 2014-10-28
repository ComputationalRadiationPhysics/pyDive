import pyDive
import numpy as np
import pytest

@pytest.fixture()
def init_pyDive(request):
    pyDive.init("mpi")

def test_particles2mesh(init_pyDive):
    shape = [256, 256]
    density = pyDive.cloned.zeros(shape)

    h5input = "sample.h5"

    particles = pyDive.h5.fromPath(h5input, "/particles")

    def particles2density(particles, density):
        total_pos = particles["cellidx"].astype(np.float32) + particles["pos"]

        # convert total_pos to an (N, 2) shaped array
        total_pos = np.hstack((total_pos["x"][:,np.newaxis],
                               total_pos["y"][:,np.newaxis]))

        par_weighting = np.ones(particles.shape)
        import pyDive.mappings
        pyDive.mappings.particles2mesh(density, par_weighting, total_pos, pyDive.mappings.CIC)

    pyDive.map(particles2density, particles, density)

    test_density = density.sum() # add up all local copies

    ref_density = np.load("p2m_CIC.npy")

    assert np.array_equal(ref_density, test_density)

def test_mesh2particles(init_pyDive):
    h5input = "sample.h5"

    particles = pyDive.h5.fromPath(h5input, "/particles")[:]
    field = pyDive.h5.fromPath(h5input, "/fields/fieldB/z")[:].gather()

    field_strengths = pyDive.empty(particles.shape)

    @pyDive.map
    def mesh2particles(field_strengths, particles, field):
        total_pos = particles["cellidx"].astype(np.float32) + particles["pos"]

        # convert total_pos to an (N, 2) shaped array
        total_pos = np.hstack((total_pos["x"][:,np.newaxis],
                               total_pos["y"][:,np.newaxis]))

        import pyDive.mappings
        field_strengths[:] = pyDive.mappings.mesh2particles(field, total_pos, pyDive.mappings.CIC)

    mesh2particles(field_strengths, particles, field=field)

    ref_field_strengths = np.load("m2p_CIC.npy")

    assert np.array_equal(ref_field_strengths, field_strengths.gather())

