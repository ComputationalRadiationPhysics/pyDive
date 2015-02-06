import pyDive
import numpy as np

def test_arrayOfStructs(init_pyDive):
    fieldE_x = np.random.rand(100, 100)
    fieldE_y = np.random.rand(100, 100)
    fieldB_z = np.random.rand(100, 100)

    fields = {"fieldE": {"x" : fieldE_x, "y" : fieldE_y}, "fieldB" : {"z" : fieldB_z}}
    fields = pyDive.arrayOfStructs(fields)

    assert np.array_equal(fieldB_z, fields["fieldB/z"])
    assert np.array_equal(fieldB_z, fields["fieldB"]["z"])
    assert np.array_equal(fieldB_z, fields["fieldB"][:]["z"])
    assert np.array_equal(fieldB_z, fields[:]["fieldB/z"])
    assert np.array_equal(fieldB_z**2, (fields**2)["fieldB/z"])
    assert np.array_equal(fieldB_z**2, (fields**2).fieldB.z)