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

class ProxyArray(object):

    def __init__(self):
        self.deltas = []

    def __getitem__(self, slice_objects):
        result = ghost()
        result.src = slice_objects
        return result

    def __setitem__(self, dst, other):
        src = other.src

        shape = [100] * len(src)
        src = [slice(*s.indices(sh)) for s, sh in zip(src, shape)]
        dst = [slice(*d.indices(sh)) for d, sh in zip(dst, shape)]

        self.deltas.append([s.start - d.start for s, d in zip(src, dst)])