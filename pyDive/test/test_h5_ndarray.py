import pyDive
import numpy as np
import random
import h5py as h5
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.h5")

def test_h5_ndarray1(init_pyDive):
    dataset = "particles/pos/x"

    test_array = pyDive.h5.open(input_file, dataset)

    ref_array = h5.File(input_file, "r")[dataset][:]

    assert np.array_equal(ref_array, test_array.load().gather())

def test_h5_ndarray2(init_pyDive):
    window = np.s_[31:201:3, 2:233:5]
    dataset = "/fields/fieldE/x"

    test_array = pyDive.h5.open(input_file, dataset, distaxis=1)

    ref_array = h5.File(input_file, "r")[dataset][window]

    assert np.array_equal(ref_array, test_array[window].load().gather())

def test_h5(init_pyDive):
    test_array = pyDive.h5.open(input_file, "particles/pos")

    ref_array_x = h5.File(input_file, "r")["particles/pos/x"][:]
    ref_array_y = h5.File(input_file, "r")["particles/pos/y"][:]

    assert np.array_equal(ref_array_x, test_array["x"].load().gather())
    assert np.array_equal(ref_array_y, test_array["y"].load().gather())
    assert np.array_equal(ref_array_x, test_array.load()["x"].gather())
    assert np.array_equal(ref_array_y, test_array.load()["y"].gather())
