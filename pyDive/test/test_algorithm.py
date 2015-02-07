import pyDive
import numpy as np
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.h5")

def test_map(init_pyDive):
    input_array = pyDive.h5.open(input_file, "fields")

    ref_array = input_array["fieldE/x"].load().gather()**2 \
              + input_array["fieldE/y"].load().gather()**2 \
              + input_array["fieldB/z"].load().gather()**2

    test_array = pyDive.empty(input_array.shape, dtype=input_array.dtype["fieldE"]["x"])

    def energy(out, h5fields):
        fields = h5fields.load()
        out[:] = fields["fieldE/x"]**2 + fields["fieldE/y"]**2 + fields["fieldB/z"]**2

    pyDive.map(energy, test_array, input_array)

    assert np.array_equal(ref_array, test_array.gather())

def test_reduce(init_pyDive):
    input_array = pyDive.h5.open(input_file, "fields").load()

    energy_array = pyDive.empty(input_array.shape, dtype=input_array.dtype["fieldE"]["x"])

    def energy(out, fields):
        out[:] = fields["fieldE/x"]**2 + fields["fieldE/y"]**2 + fields["fieldB/z"]**2

    pyDive.map(energy, energy_array, input_array)

    test_total = pyDive.reduce(energy_array, np.add)
    ref_total = np.add.reduce(energy_array.gather(), axis=None)

    diff = abs(ref_total - test_total)
    assert diff / ref_total < 1.0e-5

def test_mapReduce(init_pyDive):
    input_array = pyDive.h5.open(input_file, "fields")

    ref_array = input_array["fieldE/x"].load().gather()**2 \
              + input_array["fieldE/y"].load().gather()**2 \
              + input_array["fieldB/z"].load().gather()**2
    ref_total = np.add.reduce(ref_array, axis=None)

    test_total = pyDive.mapReduce(\
        lambda fields: fields["fieldE/x"].load()**2 + fields["fieldE/y"].load()**2 + fields["fieldB/z"].load()**2,
        np.add, input_array)

    diff = abs(ref_total - test_total)
    assert diff / ref_total < 1.0e-5