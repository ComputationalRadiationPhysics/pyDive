def getFirstSubIdx(slice_obj, begin, end):
    if slice_obj.start > begin:
        if slice_obj.start >= end: return None
        return slice_obj.start
    i = (begin-1 - slice_obj.start) / slice_obj.step + 1
    idx = slice_obj.start + i * slice_obj.step
    if idx >= end or idx >= slice_obj.stop: return None
    return idx

def sliceShape(slices, shape):
    new_shape = list(shape)
    for i in range(len(slices)):
        if type(slices[i]) is int:
            new_shape[i] = 1
            continue
        # create a clean, wrapped slice object
        wrapped_ids = slices[i].indices(shape[i])
        clean_slice = slice(*wrapped_ids)
        # new size of axis i
        new_shape[i] = (clean_slice.stop-1 - clean_slice.start) / clean_slice.step + 1
    return new_shape