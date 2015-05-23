"""
Copyright 2015 Heiko Burau

This file is part of pyDive.

pyDive is free software: you can redistribute it and/or modify
it under the terms of of either the GNU General Public License or
the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
pyDive is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License and the GNU Lesser General Public License
for more details.

You should have received a copy of the GNU General Public License
and the GNU Lesser General Public License along with pyDive.
If not, see <http://www.gnu.org/licenses/>.
"""
__doc__ = None
import numpy as np
from mpi4py import MPI

def scatterArrayMPI_async(in_array, commData, target2rank):
    send_bufs = [] # todo: check if necessary
    tasks = []
    for (dest_target, window, tag) in commData:
        send_bufs.append(np.empty_like(in_array[window]))
        send_bufs[-1][:] = in_array[window]
        tasks.append(MPI.COMM_WORLD.Isend(send_bufs[-1], dest=target2rank[dest_target], tag=tag))

    return tasks

def gatherArraysMPI_async(out_array, commData, target2rank):
    recv_bufs = []
    tasks = []
    for (src_target, window, tag) in commData:
        recv_bufs.append(np.empty_like(out_array[window]))
        tasks.append(MPI.COMM_WORLD.Irecv(recv_bufs[-1], source=target2rank[src_target], tag=tag))

    return tasks, recv_bufs

def finish_communication(out_array, commData, recv_bufs):
    for (src_target, window, tag), recv_buf in zip(commData, recv_bufs):
        out_array[window] = recv_buf

import os
onTarget = os.environ.get("onTarget", 'False')

# execute this code only if it is not executed on engine
if onTarget == 'False':
    import pyDive.IPParallelClient as com

    def MPI_copier(source, dest):
        view = com.getView()

        # send
        view.execute('{0}_send_tasks = interengine.scatterArrayMPI_async({0}, src_commData[0], target2rank)'\
            .format(source.name), targets=source.target_ranks)

        # receive
        view.execute("""\
            {0}_recv_tasks, {0}_recv_bufs = interengine.gatherArraysMPI_async({1}, dest_commData[0], target2rank)
            """.format(source.name, dest.name),\
            targets=dest.target_ranks)

        # finish communication
        view.execute('''\
            if "{0}_send_tasks" in locals():
                MPI.Request.Waitall({0}_send_tasks)
                del {0}_send_tasks
            if "{0}_recv_tasks" in locals():
                MPI.Request.Waitall({0}_recv_tasks)
                interengine.finish_communication({1}, dest_commData[0], {0}_recv_bufs)
                del {0}_recv_tasks, {0}_recv_bufs
            '''.format(source.name, dest.name),
            targets=tuple(set(source.target_ranks + dest.target_ranks)))