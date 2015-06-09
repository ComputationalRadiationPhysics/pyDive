# -*- coding: utf-8 -*-

import pyDive
import numpy as np
import random
import pytest

gpu_enabled = pytest.mark.skipif(not hasattr(pyDive, "gpu"), reason="pycuda not installed")

sizes = ((1,), (5,), (29,), (64,),
        (1, 1), (1, 5), (5, 29), (64, 1), (64, 64),
        (1, 1, 1), (8, 8, 8), (1, 2, 3), (12, 37, 50), (64,64,64))
dtypes = (np.float,)

@gpu_enabled
def test_basics(init_pyDive):
    for size in sizes:
        for dtype in dtypes:
            ref = (np.random.rand(*size) * 100.0).astype(dtype)
            test_array_cpu = pyDive.array(ref)
            test_array = pyDive.gpu.empty(size, dtype, distaxes=range(len(size)))

            test_array[:] = test_array_cpu

            if all(s > 3 for s in size):
                slices = [slice(1, -2, None) for i in range(len(size))]

                a = test_array[slices]
                a = a + 1
                a = a**2
                a = a + a
                test_array[slices] = a

                b = ref[slices]
                b = b + 1
                b = b**2
                b = b + b
                ref[slices] = b

                np.testing.assert_array_almost_equal(test_array.to_cpu().gather(), ref)

@gpu_enabled
def test_interengine(init_pyDive):
    for size in sizes:
        for dtype in dtypes:
            ref = (np.random.rand(*size) * 100.0).astype(dtype)
            test_array_cpu = pyDive.array(ref)
            test_array = pyDive.gpu.empty(size, dtype, distaxes=range(len(size)))

            test_array[:] = test_array_cpu

            for i in range(len(size)):
                if size[i] < 5: continue

                slicesA = [slice(None)] * len(size)
                slicesB = list(slicesA)
                slicesA[i] = slice(0, 5)
                slicesB[i] = slice(-5, None)

                test_array[slicesA] = test_array[slicesB]
                ref[slicesA] = ref[slicesB]

                assert np.array_equal(test_array.to_cpu().gather(), ref)

                slicesA = [s/2 for s in size]
                slicesB = list(slicesA)
                slicesA[i] = slice(0, 5)
                slicesB[i] = slice(-5, None)

                test_array[slicesA] = test_array[slicesB]
                ref[slicesA] = ref[slicesB]

                assert np.array_equal(test_array.to_cpu().gather(), ref)