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

@gpu_enabled
def test_misc(init_pyDive):
    sizes = ((10,20,30), (30,20,10), (16,16,16), (16,32,48), (13,29,37))

    def do_funny_stuff(a, b):
        a[1:,1:,1:] = b[:-1,:-1,:-1]
        b[1:,1:,1:] = a[1:,:-1,:-1]
        a[1:,1:,1:] = b[1:,1:,:-1]
        b[1:,1:,1:] = a[1:,:-1,1:]
        a[:-1,:-3,:-4] = b[1:,3:,4:]
        b[0,:,0] += a[-2,:,-2]
        a[4:-3,2:-1,5:-2] = b[4:-3,2:-1,5:-2]
        b[1:3,1:4,1:5] *= a[-3:-1,-4:-1,-5:-1]
        #a[1,2,3] -= b[3,2,1]
        b[:,:,0] = a[:,:,1]

    for size in sizes:
        cpu_a = (np.random.rand(*size) * 100.0).astype(np.int)
        cpu_b = (np.random.rand(*size) * 100.0).astype(np.int)

        gpu_a = pyDive.gpu.array(cpu_a)
        gpu_b = pyDive.gpu.array(cpu_b)

        do_funny_stuff(cpu_a, cpu_b)
        do_funny_stuff(gpu_a, gpu_b)

        assert np.array_equal(gpu_a.to_cpu(), cpu_a)
        assert np.array_equal(gpu_b.to_cpu(), cpu_b)










