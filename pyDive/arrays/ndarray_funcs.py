# Copyright 2016 Heiko Burau
#
# This file is part of pyDive.
#
# pyDive is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pyDive.  If not, see <http://www.gnu.org/licenses/>.

import pyDive
import numpy as np


def mean(a):
    return np.mean(np.array(pyDive.map(lambda a: np.mean(a), a)))


def amin(a):
    return pyDive.reduce(min, a, np.amin)


def amax(a):
    return pyDive.reduce(max, a, np.amax)
