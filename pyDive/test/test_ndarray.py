import pyDive
import numpy as np
import random

sizes = ((1,), (5,), (29,), (64,),
        (1, 1), (1, 5), (5, 29), (64, 1), (64, 64),
        (1, 1, 1), (8, 8, 8), (1, 2, 3), (12, 37, 50), (64,64,64))
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

                # bitmask indexing
                bitmask = pyDive.array(np.random.rand(*size) > 0.5, distaxes=test_array.distaxes)
                assert ref[bitmask.gather()].shape == test_array[bitmask].shape
                # ordering can be distinct, thus merely check if sets are equal
                assert set(ref[bitmask.gather()]) == set(test_array[bitmask].gather())

                ref2 = ref.copy()
                test_array2 = test_array.copy()
                ref2[bitmask.gather()] = 1
                test_array2[bitmask] = 1
                assert np.array_equal(ref2, test_array2.gather())

def test_multiple_axes(init_pyDive):
    for size in sizes:
        for dtype in dtypes:
            ref = (np.random.rand(*size) * 100.0).astype(dtype)

            for distaxes in [range(i+1) for i in range(len(size))]:
                test_array = pyDive.empty(size, dtype, distaxes)
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

                # bitmask indexing
                bitmask = pyDive.array(np.random.rand(*size) > 0.5, distaxes=test_array.distaxes)
                assert ref[bitmask.gather()].shape == test_array[bitmask].shape
                # ordering can be distinct, thus merely check if sets are equal
                assert set(ref[bitmask.gather()]) == set(test_array[bitmask].gather())

                ref2 = ref.copy()
                test_array2 = test_array.copy()
                ref2[bitmask.gather()] = 1
                test_array2[bitmask] = 1
                assert np.array_equal(ref2, test_array2.gather())

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

                test_array[slicesA] = test_array[slicesB]
                ref[slicesA] = ref[slicesB]

                assert np.array_equal(test_array.gather(), ref)

                slicesA = [s/2 for s in size]
                slicesB = list(slicesA)
                slicesA[distaxis] = slice(0, 5)
                slicesB[distaxis] = slice(-5, None)

                test_array[slicesA] = test_array[slicesB]
                ref[slicesA] = ref[slicesB]

                assert np.array_equal(test_array.gather(), ref)

def test_interengine_multiple_axes(init_pyDive):
    for size in sizes:
        for dtype in dtypes:
            ref = (np.random.rand(*size) * 100.0).astype(dtype)

            for distaxesA in [range(i+1) for i in range(len(size))]:
                for distaxesB in [range(i,len(size)) for i in range(len(size))]:
                    test_arrayA = pyDive.empty(size, dtype, distaxesA)
                    test_arrayB = pyDive.empty(size, dtype, distaxesB)

                    test_arrayA[:] = ref
                    test_arrayB[:] = ref

                    for distaxis in range(len(size)):
                        if size[distaxis] < 5: continue

                        slicesA = [slice(None)] * len(size)
                        slicesB = list(slicesA)
                        slicesA[distaxis] = slice(0, 5)
                        slicesB[distaxis] = slice(-5, None)

                        ref_sum = ref[slicesA] + ref[slicesB]
                        test_array_sum = test_arrayA[slicesA] + test_arrayB[slicesB]

                        assert np.array_equal(ref_sum, test_array_sum.gather())

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
        a[1,2,3] -= b[3,2,1]
        b[:,:,0] = a[:,:,1]

    for size in sizes:
        print "size", size
        np_a = (np.random.rand(*size) * 100.0).astype(np.int)
        np_b = (np.random.rand(*size) * 100.0).astype(np.int)

        pd_a = pyDive.array(np_a, distaxes=(0,1,2))
        pd_b = pyDive.array(np_b, distaxes=(0,1,2))

        do_funny_stuff(np_a, np_b)
        do_funny_stuff(pd_a, pd_b)

        assert np.array_equal(pd_a, np_a)
        assert np.array_equal(pd_b, np_b)