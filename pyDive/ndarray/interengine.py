"""
Copyright 2014 Heiko Burau

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

def gatherArraysMPI_sync(out_array, targets, tags, distaxis_sizes, distaxis, target2rank):
    window = [slice(None)] * len(shape(out_array))
    tasks = []
    pos = 0
    for i in range(len(targets)):
        window[distaxis] = slice(pos, pos + distaxis_sizes[i])
        tasks.append(MPI.COMM_WORLD.Recv(out_array[window], source=target2rank[targets[i]], tag=tags[i]))
        pos += distaxis_sizes[i]

def scatterArrayMPI_async(in_array, targets, tags, distaxis_sizes, distaxis, target2rank):
    window = [slice(None)] * len(shape(in_array))
    pos = 0
    for i in range(len(targets)):
        window[distaxis] = slice(pos, pos + distaxis_sizes[i])
        tmp = np.empty_like(in_array[window])
        tmp[:] = in_array[window]
        MPI.COMM_WORLD.Isend(tmp, dest=target2rank[targets[i]], tag=tags[i])
        pos += distaxis_sizes[i]