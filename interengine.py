def gatherArraysMPI_sync(out_array, targets, tags, distaxis_sizes, distaxis, target2rank):
    window = [slice(None)] * len(shape(out_array))
    tasks = []
    pos = 0
    for i in range(len(targets)):
        window[distaxis] = slice(pos, pos + distaxis_sizes[i])
        tasks.append(MPI.COMM_WORLD.Recv(out_array[window], source=target2rank[targets[i]], tag=tags[i]))
        pos += distaxis_sizes[i]

    #for task in tasks:
        #task.wait()

def scatterArrayMPI_async(in_array, targets, tags, distaxis_sizes, distaxis, target2rank):
    window = [slice(None)] * len(shape(in_array))
    pos = 0
    for i in range(len(targets)):
        window[distaxis] = slice(pos, pos + distaxis_sizes[i])
        tmp = empty_like(in_array[window])
        tmp[:] = in_array[window]
        MPI.COMM_WORLD.Isend(tmp, dest=target2rank[targets[i]], tag=tags[i])
        pos += distaxis_sizes[i]