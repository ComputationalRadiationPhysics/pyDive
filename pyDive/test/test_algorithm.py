import pyDive
import numpy as np
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.h5")


def test_map(init_pyDive):
    input_array = pyDive.h5.open(input_file, "fields")

    ref_array =\
        input_array["fieldE/x"].load().gather()**2 +\
        input_array["fieldE/y"].load().gather()**2 +\
        input_array["fieldB/z"].load().gather()**2

    def energy(h5fields):
        fields = h5fields.load()
        return fields["fieldE/x"]**2 + fields["fieldE/y"]**2 + fields["fieldB/z"]**2

    test_array = pyDive.map(energy, input_array)

    assert np.array_equal(ref_array, test_array.gather())


def test_reduce(init_pyDive):
    input_array = pyDive.h5.open(input_file, "fields").load()

    def energy(fields):
        return fields["fieldE/x"]**2 + fields["fieldE/y"]**2 + fields["fieldB/z"]**2

    energy_array = pyDive.map(energy, input_array)

    test_total = pyDive.reduce(np.add, energy_array)
    ref_total = np.add.reduce(energy_array.gather(), axis=None)

    diff = abs(ref_total - test_total)
    assert diff / ref_total < 1.0e-5
