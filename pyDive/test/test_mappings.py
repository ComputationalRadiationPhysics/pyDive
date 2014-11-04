import pyDive
import numpy as np
import os

dirname = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(dirname, "sample.h5")

def test_particles2mesh(init_pyDive):
    shape = [256, 256]
    density = pyDive.cloned.zeros(shape)

    particles = pyDive.h5.fromPath(input_file, "/particles")

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

    ref_density = np.load(os.path.join(dirname, "p2m_CIC.npy"))

    import matplotlib.pyplot as plt
    plt.imshow(ref_density - test_density)
    plt.colorbar()
    plt.savefig("diff.png")
    plt.clf()
    plt.imshow(ref_density)
    plt.colorbar()
    plt.savefig("ref_density.png")
    plt.clf()
    plt.imshow(test_density)
    plt.colorbar()
    plt.savefig("test_density.png")
    plt.clf()

    np.testing.assert_array_almost_equal(ref_density, test_density)

def test_mesh2particles(init_pyDive):
    particles = pyDive.h5.fromPath(input_file, "/particles")[:]
    field = pyDive.h5.fromPath(input_file, "/fields/fieldB/z")[:].gather()

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

    ref_field_strengths = np.load(os.path.join(dirname, "m2p_CIC.npy"))

    assert np.array_equal(ref_field_strengths, field_strengths.gather())

