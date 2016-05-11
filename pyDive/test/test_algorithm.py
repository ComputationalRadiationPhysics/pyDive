import pyDive
import numpy as np
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.h5")


def test_map(init_pyDive):
    input_array_x = pyDive.h5.open(input_file, "fields/fieldE/x")
    input_array_y = pyDive.h5.open(input_file, "fields/fieldE/y")
    input_array_z = pyDive.h5.open(input_file, "fields/fieldB/z")

    ref_array =\
        input_array_x.load().gather()**2 +\
        input_array_y.load().gather()**2 +\
        input_array_z.load().gather()**2

    def energy(h5field_x, h5field_y, h5field_z):
        field_x = h5field_x.load()
        field_y = h5field_y.load()
        field_z = h5field_z.load()
        return field_x**2 + field_y**2 + field_z**2

    test_array = pyDive.map(energy, input_array_x, input_array_y, input_array_z)

    assert np.array_equal(ref_array, test_array)


def test_reduce(init_pyDive):
    input_array_x = pyDive.h5.open(input_file, "fields/fieldE/x")
    input_array_y = pyDive.h5.open(input_file, "fields/fieldE/y")
    input_array_z = pyDive.h5.open(input_file, "fields/fieldB/z")

    def energy(h5field_x, h5field_y, h5field_z):
        field_x = h5field_x.load()
        field_y = h5field_y.load()
        field_z = h5field_z.load()
        return field_x**2 + field_y**2 + field_z**2

    energy_array = pyDive.map(energy, input_array_x, input_array_y, input_array_z)

    test_total = pyDive.reduce(np.add, energy_array)
    ref_total = np.add.reduce(energy_array.gather(), axis=None)

    diff = abs(ref_total - test_total)
    assert diff / ref_total < 1.0e-5
