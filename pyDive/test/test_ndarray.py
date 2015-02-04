import pyDive
import numpy as np
import random

sizes = ((1,), (5,), (29,), (64,),
        (1, 1), (1, 5), (5, 29), (64, 1), (64, 64),
        (1, 1, 1), (8, 8, 8), (1, 2, 3), (12, 37, 50))
dtypes = (np.int,)

def test_slicing(init_pyDive):
    for size in sizes:
        for dtype in dtypes:
            ref = (np.random.rand(*size) * 100.0).astype(dtype)

            for distaxis in range(len(size)):
                test_array = pyDive.empty(size, dtype, distaxis)
                test_array[:] = ref

                slices = []
                for i in range(len(size)):
                    start = size[i] / 3
                    stop = size[i] - size[i] / 5
                    step = 2
                    slices.append(slice(start, stop, step))

                assert np.array_equal(ref[slices], test_array[slices].gather())

                slices = []
                for i in range(len(size)):
                    slices.append(slice(-5, None, None))

                assert np.array_equal(ref[slices], test_array[slices].gather())

                slices = []
                for i in range(len(size)):
                    slices.append(slice(0, 5, None))

                assert np.array_equal(ref[slices], test_array[slices].gather())


def test_interengine(init_pyDive):
    for size in sizes:
        for dtype in dtypes:
            ref = (np.random.rand(*size) * 100.0).astype(dtype)

            for distaxis in range(len(size)):
                if size[distaxis] < 5: continue

                test_array = pyDive.empty(size, dtype, distaxis)
                test_array[:] = ref

                slicesA = [slice(None)] * len(size)
                slicesB = list(slicesA)
                slicesA[distaxis] = slice(0, 5)
                slicesB[distaxis] = slice(-5, None)

                print size, distaxis

                ref_sum = ref[slicesA] + ref[slicesB]
                test_array_sum = test_array[slicesA] + test_array[slicesB]

                assert np.array_equal(ref_sum, test_array_sum.gather())