# -*- coding: utf-8 -*-

import pyDive
import numpy as np
import pytest
import os

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adios_test.bp")

adios_enabled = pytest.mark.skipif(not hasattr(pyDive, "adios"), reason="adios not installed")


@adios_enabled
def test_adios1(init_pyDive):
    import adios

    var_path = "temperature"

    fileHandle = adios.file(input_file)
    ref = fileHandle.var[var_path].read()
    fileHandle.close()

    test = pyDive.adios.open(input_file, var_path)

    assert np.array_equal(test.load(), ref)
