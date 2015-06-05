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

import pycuda.gpuarray
import pycuda.driver as cuda
import numpy as np

class gpu_ndarray(pycuda.gpuarray.GPUArray):

    def __init__(self, shape, dtype, allocator=cuda.mem_alloc,\
                    base=None, gpudata=None, strides=None, order="C"):
        super(gpu_ndarray, self).__init__(shape, dtype, allocator, base, gpudata, strides, order)

    def __setitem__(self, key, other):
        # if args is [:] then assign other to the entire ndarray
        if key == slice(None):
            if len(self.shape) == 1:
                if self.flags.forc and other.flags.forc:
                    # both arrays are a contiguous block of memory
                    cuda.memcpy_dtod(self.gpudata, other.gpudata, self.nbytes)
                else:
                    copy = cuda.Memcpy2D()
                    copy.set_src_device(self.gpudata)
                    copy.src_pitch = self.strides[0]
                    copy.set_dst_device(other.gpudata)
                    copy.dst_pitch = other.strides[0]
                    copy.width_in_bytes = np.dtype(self.dtype).itemsize
                    copy.height = self.shape[0]
                    copy(aligned=False)

                return
            raise RuntimeError("item assignment is only supported for 1D arrays")

    def __cast_from_base(self, pycuda_array):
        return gpu_ndarray(pycuda_array.shape, pycuda_array.dtype, pycuda_array.allocator,\
            pycuda_array.base, pycuda_array.gpudata, pycuda_array.strides)

    def __getitem__(self, args):
        pycuda_array = super(gpu_ndarray, self).__getitem__(tuple(args))
        return self.__cast_from_base(pycuda_array)

    def to_cpu(self):
        if self.flags.forc:
            return self.get(pagelocked=True)

        result = cuda.pagelocked_empty(self.shape, self.dtype)
        itemsize = np.dtype(self.dtype).itemsize
        if len(self.shape) == 1:
            copy = cuda.Memcpy2D()
            copy.set_src_device(self.gpudata)
            copy.src_pitch = self.strides[0]
            copy.set_dst_host(result)
            copy.dst_pitch = itemsize
            copy.width_in_bytes = itemsize
            copy.height = self.shape[0]
            copy(aligned=False)
            return result

        elif len(self.shape) == 2:
            if itemsize == self.strides[1]:
                # contiguous block of memory for each row
                copy = cuda.Memcpy2D()
                copy.set_src_device(self.gpudata)
                copy.src_pitch = self.strides[0]
                copy.set_dst_host(result)
                copy.dst_pitch = itemsize * result.shape[1]
                copy.width_in_bytes = copy.dst_pitch
                copy.height = self.shape[0]
                copy(aligned=False)
                return result
            else:
                # array has to copied column by column, because there a two different pitches
                # which is not supported by cuda.
                copy = cuda.Memcpy2D()
                copy.set_src_device(self.gpudata)
                copy.src_pitch = self.strides[0]
                copy.set_dst_host(result)
                copy.dst_pitch = itemsize
                copy.width_in_bytes = itemsize
                copy.height = self.shape[0]

                for col in range(self.shape[1]):
                    copy.src_x_in_bytes = col * self.strides[1]
                    copy(aligned=False)
                return result
        elif len(self.shape) == 3:
            copy = cuda.Memcpy3D()
            copy.set_src_device(self.gpudata)
            copy.src_pitch = self.strides[1]
            copy.set_dst_host(result)
            copy.dst_pitch = itemsize * result.shape[2]
            copy.width_in_bytes = copy.dst_pitch
            copy.height = self.shape[1]
            copy.depth = self.shape[0]
            copy()
            return result
        else:
            raise RuntimeError("dimension %d is not supported." % len(self.shape))

    def __elementwise_op__(self, op, *args):
        pycuda_array = getattr(super(gpu_ndarray, self), op)(*args)
        return self.__cast_from_base(pycuda_array)

    def __elementwise_iop__(self, op, *args):
        getattr(super(gpu_ndarray, self), op)(*args)
        return self

# add special operations like __add__, __mul__, etc. to `gpu_ndarray`

binary_ops = ["add", "sub", "mul", "floordiv", "div", "mod", "pow", "lshift", "rshift", "and", "xor", "or"]

binary_iops = ["__i" + op + "__" for op in binary_ops]
binary_rops = ["__r" + op + "__" for op in binary_ops]
binary_ops = ["__" + op + "__" for op in binary_ops]
unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__long__", "__float__", "__oct__", "__hex__"]
comp_ops = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]

special_ops_avail = set(name for name in pycuda.gpuarray.GPUArray.__dict__.keys() if name.endswith("__"))

make_special_op = lambda op: lambda self, *args: self.__elementwise_op__(op, *args)
make_special_iop = lambda op: lambda self, *args: self.__elementwise_iop__(op, *args)

special_ops_dict = {op : make_special_op(op) for op in \
    set(binary_ops + binary_rops + unary_ops + comp_ops) & special_ops_avail}
special_iops_dict = {op : make_special_iop(op) for op in set(binary_iops) & special_ops_avail}

from types import MethodType

for name, func in special_ops_dict.items() + special_iops_dict.items():
    setattr(gpu_ndarray, name, MethodType(func, None, gpu_ndarray))

# -------------------- factories -----------------------------------

# get a `gpu_ndarray` instance out of a `pycuda.gpuarray.GPUArray` instance
def gpu_ndarray_cast(pycuda_array):
    return gpu_ndarray(pycuda_array.shape, pycuda_array.dtype, pycuda_array.allocator,\
        pycuda_array.base, pycuda_array.gpudata, pycuda_array.strides)

def empty(shape, dtype):
    return gpu_ndarray(shape, dtype)

pycuda_factories = ("zeros", "empty_like", "zeros_like")

# wrap all factory functions from pycuda so that they return a `gpu_ndarray` instance
from functools import wraps
make_factory = lambda func : wraps(func)(lambda *args, **kwargs: gpu_ndarray_cast(func(*args, **kwargs)))
factories = {func : make_factory(getattr(pycuda.gpuarray, func)) for func in pycuda_factories}

globals().update(factories)
