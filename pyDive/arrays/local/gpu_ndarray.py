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

def copy_non_contiguous(dst, src):
    """Copy ``src`` array to ``dst`` array. A gpu-array may have a non contiguous block of memory,
    i.e. it may have substancial pitches/strides. However a cpu-array must have a contiguous block of memory.
    All four directions are allowed.
    """
    assert src.dtype == dst.dtype,\
        "src ({}) and dst ({}) must have the same datatype.".format(str(src.dtype), str(dst.dtype))
    assert dst.shape == src.shape,\
        "Shapes do not match: " + str(dst.shape) + " <-> " + str(src.shape)

    itemsize = np.dtype(src.dtype).itemsize
    copy = cuda.Memcpy2D()
    src_on_gpu = isinstance(src, pycuda.gpuarray.GPUArray)
    dst_on_gpu = isinstance(dst, pycuda.gpuarray.GPUArray)
    if src_on_gpu:
        copy.set_src_device(src.gpudata)
    else:
        copy.set_src_host(src)
    if dst_on_gpu:
        copy.set_dst_device(dst.gpudata)
    else:
        copy.set_dst_host(dst)

    if len(src.shape) == 1:
        copy.src_pitch = src.strides[0] if src_on_gpu else itemsize
        copy.dst_pitch = dst.strides[0] if dst_on_gpu else itemsize
        copy.width_in_bytes = itemsize
        copy.height = src.shape[0]
        copy(aligned=False)

    elif len(src.shape) == 2:
        if (itemsize != src.strides[1] if src_on_gpu else False) or \
           (itemsize != dst.strides[1] if dst_on_gpu else False):
            # arrays have to be copied column by column, because there a two substantial pitches/strides
            # which is not supported by cuda.
            copy.src_pitch = src.strides[0] if src_on_gpu else itemsize
            copy.dst_pitch = dst.strides[0] if dst_on_gpu else itemsize
            copy.width_in_bytes = itemsize
            copy.height = src.shape[0]

            for col in range(src.shape[1]):
                copy.src_x_in_bytes = col * src.strides[1] if src_on_gpu else col * itemsize
                copy.dst_x_in_bytes = col * dst.strides[1] if dst_on_gpu else col * itemsize
                copy(aligned=False)
        else:
            # both arrays have a contiguous block of memory for each row
            copy.src_pitch = src.strides[0] if src_on_gpu else itemsize * src.shape[1]
            copy.dst_pitch = dst.strides[0] if dst_on_gpu else itemsize * src.shape[1]
            copy.width_in_bytes = itemsize * src.shape[1]
            copy.height = src.shape[0]
            copy(aligned=False)

    elif len(src.shape) == 3:
        if (src.strides[0] != src.shape[1] * src.strides[1] if src_on_gpu else False) or \
           (dst.strides[0] != dst.shape[1] * dst.strides[1] if dst_on_gpu else False):
            # arrays have to be copied plane by plane, because there a substantial pitche/stride
            # for the z-axis which is not supported by cuda.
            for plane in range(src.shape[0]):
                copy_non_contiguous(dst[plane,:,:], src[plane,:,:])
            return

        copy = cuda.Memcpy3D()
        if src_on_gpu:
            copy.set_src_device(src.gpudata)
        else:
            copy.set_src_host(src)
        if dst_on_gpu:
            copy.set_dst_device(dst.gpudata)
        else:
            copy.set_dst_host(dst)

        copy.src_pitch = src.strides[1] if src_on_gpu else itemsize * src.shape[2]
        copy.dst_pitch = dst.strides[1] if dst_on_gpu else itemsize * src.shape[2]
        copy.width_in_bytes = itemsize * src.shape[2]
        copy.height = copy.src_height = copy.dst_height = src.shape[1]
        copy.depth = src.shape[0]

        copy()
    else:
        raise RuntimeError("dimension %d is not supported." % len(src.shape))


class gpu_ndarray(pycuda.gpuarray.GPUArray):

    def __init__(self, shape, dtype, allocator=cuda.mem_alloc,\
                    base=None, gpudata=None, strides=None, order="C"):
        if type(shape) not in (list, tuple):
            shape = (shape,)
        elif type(shape) is not tuple:
            shape = tuple(shape)
        super(gpu_ndarray, self).__init__(shape, np.dtype(dtype), allocator, base, gpudata, strides, order)

    def __setitem__(self, key, other):
        # if args is [:] then assign `other` to the entire ndarray
        if key == slice(None):
            if isinstance(other, pycuda.gpuarray.GPUArray):
                if self.flags.forc and other.flags.forc:
                    # both arrays are a contiguous block of memory
                    cuda.memcpy_dtod(self.gpudata, other.gpudata, self.nbytes)
                    return
            else:
                if self.flags.forc:
                    # both arrays are a contiguous block of memory
                    cuda.memcpy_htod(self.gpudata, other)
                    return

            copy_non_contiguous(self, other)
            return

        # assign `other` to sub-array of self
        sub_array = self[key]
        sub_array[:] = other

    def __cast_from_base(self, pycuda_array):
        return gpu_ndarray(pycuda_array.shape, pycuda_array.dtype, pycuda_array.allocator,\
            pycuda_array.base, pycuda_array.gpudata, pycuda_array.strides)

    def __getitem__(self, args):
        pycuda_array = super(gpu_ndarray, self).__getitem__(tuple(args))
        return self.__cast_from_base(pycuda_array)

    def copy(self):
        if self.flags.forc:
            return super(gpu_ndarray, self).copy()
        result = gpu_ndarray(shape=self.shape, dtype=self.dtype, allocator=self.allocator)
        result[:] = self
        return result

    def to_cpu(self):
        if self.flags.forc:
            return self.get(pagelocked=True)

        result = cuda.pagelocked_empty(self.shape, self.dtype)
        copy_non_contiguous(result, self)
        return result

    def __elementwise_op__(self, op, *args):
        # if arrays are not contiguous make a contiguous copy
        my_array = self.copy() if not self.flags.forc else self

        args = [arg.copy() if isinstance(arg, pycuda.gpuarray.GPUArray) and not arg.flags.forc else arg for arg in args]
        pycuda_array = getattr(super(gpu_ndarray, my_array), op)(*args)
        return self.__cast_from_base(pycuda_array)

# add special operations like __add__, __mul__, etc. to `gpu_ndarray`

binary_ops = ["add", "sub", "mul", "floordiv", "div", "mod", "pow", "lshift", "rshift", "and", "xor", "or"]

binary_iops = ["__i" + op + "__" for op in binary_ops]
binary_rops = ["__r" + op + "__" for op in binary_ops]
binary_ops = ["__" + op + "__" for op in binary_ops]
unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__long__", "__float__", "__oct__", "__hex__"]
comp_ops = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]

special_ops_avail = set(name for name in pycuda.gpuarray.GPUArray.__dict__.keys() if name.endswith("__"))

make_special_op = lambda op: lambda self, *args: self.__elementwise_op__(op, *args)

special_ops_dict = {op : make_special_op(op) for op in \
    set(binary_ops + binary_rops + binary_iops + unary_ops + comp_ops) & special_ops_avail}

from types import MethodType

for name, func in special_ops_dict.items():
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
