import numpy as np
import IPParallelClient as com
#import ndarray    # this import is done by the ndarray module itself due to circular dependencies

def hollow(shape, distaxis, dtype=np.float):
    return ndarray.ndarray(shape, distaxis, dtype, no_allocation=True)

def empty(shape, distaxis, dtype=np.float):
    return ndarray.ndarray(shape, distaxis, dtype)

def hollow_like(a):
    return ndarray.ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use, no_allocation=True)

def empty_like(a):
    return ndarray.ndarray(a.shape, a.distaxis, a.dtype, a.idx_ranges, a.targets_in_use)

def array(array_like, distaxis):
    # numpy array
    if isinstance(array_like, np.ndarray):
        # result ndarray
        result = ndarray.ndarray(array_like.shape, distaxis, array_like.dtype, no_allocation=True)

        tmp = np.rollaxis(array_like, distaxis)
        sub_arrays = [tmp[begin:end] for begin, end in result.idx_ranges]
        # roll axis back
        sub_arrays = [np.rollaxis(ar, 0, distaxis+1) for ar in sub_arrays]

        view = com.getView()
        view.scatter('sub_array', sub_arrays, targets=result.targets_in_use)
        view.execute("%s = sub_array[0].copy()" % result.name, targets=result.targets_in_use)

        return result

def array_from_database(database_name, distaxis, *slice_objs):
    view = com.getView()
    db_shape, db_dtype = view.pull('%s.shape, %s.dtype' % (database_name, database_name), targets=0)

    print db_shape, db_dtype

    if not isinstance(db_shape, list) and not isinstance(db_shape, tuple):
        db_shape = [db_shape]

    assert len(db_shape) == len(slice_objs)
    # create wrapped, clean slice objects
    clean_slices = [slice(*slice_objs[i].indices(db_shape[i])) for i in range(len(slice_objs))]
    # shape of the result ndarray
    new_shape = [(slice_obj.stop-1 - slice_obj.start) / slice_obj.step + 1 for slice_obj in clean_slices]

    # result ndarray
    result = hollow(new_shape, distaxis, dtype=db_dtype)

    # create slice objects for each engine
    local_slices = [[clean_slices] for i in range(len(result.targets_in_use))]
    distaxis_slice = clean_slices[distaxis]
    for i in range(len(result.targets_in_use)):
        begin, end = result.idx_ranges[i]

        local_slices[i][distaxis] = slice(distaxis_slice.start + distaxis_slice.step * begin,\
                                          distaxis_slice.start + distaxis_slice.step * end,\
                                          distaxis_slice.step)

    # scatter slice objects to the engines
    view.scatter('args', local_slices, targets=result.targets_in_use)

    view.execute('%s = %s[tuple(args[0])]' % (result.name, database_name), targets=result.targets_in_use)
    return result