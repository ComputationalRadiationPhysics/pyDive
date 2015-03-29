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
import pyDive.IPParallelClient as com
import helper

array_id = 0

class DistributedGenericArray(object):
    """
    Represents a cluster-wide, multidimensional, homogeneous array of fixed-size elements.
    *cluster-wide* means that its elements are distributed across IPython.parallel-engines.
    The distribution is done in one dimension along a single, user-specified axis.
    The user can optionally specify which engine maps to which index range or leave the default that persuits an uniform distribution across all engines.

    This **{arraytype_name}** - class is auto-generated out of its local counterpart: **{local_arraytype_name}**.

    The implementation is based on IPython.parallel and local {local_arraytype_name} - arrays.
    Every special operation {local_arraytype_name} implements ("__add__", "__le__", ...) is also
    available for {arraytype_name}.

    Note that array slicing is a cheap operation since no memory is copied.
    However this can easily lead to the situation where you end up with two arrays of the same size but of distinct element distribution.
    Therefore call dist_like() first before doing any manual stuff on their local arrays.
    However every cluster-wide array operation first equalizes the distribution of all involved arrays,
    so an explicit call to dist_like() is rather unlikely in most use cases.
    """
    local_arraytype = None
    target_modulename = None
    interengine_copier = None
    may_allocate = True

    def __init__(self, shape, dtype=np.float, distaxis=0, target_offsets=None, target_ranks=None, no_allocation=False, **kwargs):
        """Creates an instance of {arraytype_name}. This is a low-level method of instantiating an array, it should rather be
        constructed using factory functions ("empty", "zeros", "open", ...)

        :param ints shape: shape of array
        :param dtype: datatype of a single element
        :param int distaxis: distributed axis
        :param ints target_offsets: list of indices marking the offset along the distributed axis of each local array.
        :param ints target_ranks: list of :term:`engine` ranks holding the local arrays.
        :param bool no_allocation: if ``True`` no instance of {local_arraytype_name} will be created on engine. Useful for
            manual instantiation of the local array.
        :param kwargs: additional keyword arguments are forwarded to the constructor of the local array.
        """
        #: size of the array on each axis
        self.shape = tuple(shape)
        ##: datatype of a single data value
        self.dtype = dtype
        #: axis on which memory is distributed across the :term:`engines <engine>`
        self.distaxis = distaxis
        ##: total bytes consumed by the elements of the array.
        self.nbytes = np.dtype(dtype).itemsize * np.prod(self.shape)
        self.view = com.getView()
        self.kwargs = kwargs

        assert distaxis >= 0 and distaxis < len(self.shape),\
            "distaxis ({}) has to be within [0,{})".format(distaxis, len(self.shape))

        if target_offsets is None or target_ranks is None:
            # number of available targets (engines)
            num_targets_av = len(self.view.targets if target_ranks is None else target_ranks)

            # shape of the local ndarray
            localshape = np.array(self.shape)
            localshape[distaxis] = (self.shape[distaxis] - 1) / num_targets_av + 1

            # number of occupied targets by this ndarray instance
            num_targets = (self.shape[distaxis] - 1) / localshape[distaxis] + 1

        if target_offsets is None:
            # this is the decomposition of the distributed axis
            self.target_offsets = np.arange(num_targets) * localshape[distaxis]
        else:
            self.target_offsets = np.array(target_offsets)

        if target_ranks is None:
            #: list of indices of the occupied engines
            self.target_ranks = tuple(range(num_targets))
        else:
            self.target_ranks = tuple(target_ranks)

        # generate a unique variable name used on target representing this instance
        global array_id
        #: Unique variable name of the local *array* on *engine*.
        #: Unless you are doing manual stuff on the *engines* there is no need for dealing with this attribute.
        self.name = 'dist_array' + str(array_id)
        array_id += 1

        if no_allocation:
            self.view.push({self.name : None}, targets=self.target_ranks)
        else:
            target_shapes = self.target_shapes()

            self.view.scatter('target_shape', target_shapes, targets=self.target_ranks)
            self.view.push({'kwargs' : kwargs, 'dtype' : dtype}, targets=self.target_ranks)
            self.view.execute('%s = %s(shape=target_shape[0], dtype=dtype, **kwargs)' % \
                (self.name, self.__class__.target_modulename + "." + self.__class__.local_arraytype.__name__), targets=self.target_ranks)

    def __del__(self):
        self.view.execute('del %s' % self.name, targets=self.target_ranks)

    def target_shapes(self):
        # generate a list of the local shape on each target in use
        targetshapes = []
        for i in range(len(self.target_offsets)-1):
            targetshape = np.array(self.shape)
            targetshape[self.distaxis] = self.target_offsets[i+1] - self.target_offsets[i]
            targetshapes.append(targetshape)
        targetshape = np.array(self.shape)
        targetshape[self.distaxis] = self.shape[self.distaxis] - self.target_offsets[-1]
        targetshapes.append(targetshape)

        return targetshapes

    def target_offset_vectors(self):
        # generate a list of the local offset vectors on each target in use
        target_offset_vectors = []
        for target_offset in self.target_offsets:
            target_offset_vector = [0] * len(self.shape)
            target_offset_vector[self.distaxis] = target_offset
            target_offset_vectors.append(target_offset_vector)

        return target_offset_vectors

    def __getitem__(self, args):
        if args == slice(None):
            args = (slice(None),) * len(self.shape)

        if not isinstance(args, list) and not isinstance(args, tuple):
            args = (args,)

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(args), len(self.shape))

        # shape of the new sliced ndarray
        new_shape, clean_view = helper.view_of_shape(self.shape, args)

        # if args is a list of indices then return a single data value
        if not new_shape:
            dist_idx = args[self.distaxis]
            target_idx = np.searchsorted(self.target_offsets, dist_idx, side="right") - 1
            local_idx = list(args)
            local_idx[self.distaxis] = dist_idx - self.target_offsets[target_idx]
            return self.view.pull("%s%s" % (self.name, repr(local_idx)), targets=self.target_ranks[target_idx])

        if type(clean_view[self.distaxis]) is int:
            # return local array because the distributed axis has vanished
            dist_idx = clean_view[self.distaxis]
            target_idx = np.searchsorted(self.target_offsets, dist_idx, side="right") - 1
            clean_view[self.distaxis] = dist_idx - self.target_offsets[target_idx]
            self.view.execute("sliced = %s%s" % (self.name, repr(clean_view)), targets=self.target_ranks[target_idx])
            return self.view.pull("sliced", targets=self.target_ranks[target_idx])

        # slice object in the direction of the distributed axis
        distaxis_slice = clean_view[self.distaxis]

        # determine properties of the new sliced ndarray
        new_target_offsets = []
        new_target_ranks = []
        local_slices = []
        total_ids = 0

        first_target_idx = np.searchsorted(self.target_offsets, distaxis_slice.start, side="right") - 1
        last_target_idx = np.searchsorted(self.target_offsets, distaxis_slice.stop, side="right")

        for i, target in zip(range(first_target_idx, last_target_idx),\
                             self.target_ranks[first_target_idx:last_target_idx]):
            # index range of current target
            begin = self.target_offsets[i]
            end = self.target_offsets[i+1] if i+1 < len(self.target_offsets) else self.shape[self.distaxis]
            # first slice index within [begin, end)
            firstSliceIdx = helper.getFirstSliceIdx(distaxis_slice, begin, end)
            if firstSliceIdx is None: continue
            # calculate last slice index of distaxis_slice
            tmp = (distaxis_slice.stop-1 - distaxis_slice.start) / distaxis_slice.step
            lastIdx = distaxis_slice.start + tmp * distaxis_slice.step
            # calculate last sub index within [begin,end)
            tmp = (end-1 - firstSliceIdx) / distaxis_slice.step
            lastSliceIdx = firstSliceIdx + tmp * distaxis_slice.step
            lastSliceIdx = min(lastSliceIdx, lastIdx)
            # slice object for current target
            local_slices.append(slice(firstSliceIdx - begin, lastSliceIdx+1 - begin, distaxis_slice.step))
            # number of indices remaining on the current target after slicing
            num_ids = (lastSliceIdx - firstSliceIdx) / distaxis_slice.step + 1
            # new offset for current target
            new_target_offsets.append(total_ids)
            total_ids += num_ids
            # target rank
            new_target_ranks.append(target)

        # shift distaxis by the number of vanished axes in the left of it
        new_distaxis = self.distaxis - sum(1 for arg in args[:self.distaxis] if type(arg) is int)

        # create resulting ndarray
        result = self.__class__(new_shape, self.dtype, new_distaxis, new_target_offsets, new_target_ranks, no_allocation=True, **self.kwargs)

        # remote slicing
        local_args_list = []
        for local_slice in local_slices:
            local_args = list(args)
            local_args[self.distaxis] = local_slice
            local_args_list.append(local_args)

        self.view.scatter('local_args', local_args_list, targets=result.target_ranks)
        self.view.execute('%s = %s[local_args[0]]' % (result.name, self.name), targets=result.target_ranks)

        return result

    def __setitem__(self, key, value):
        # if args is [:] then assign value to the entire ndarray
        if key == slice(None):
            if isinstance(value, self.__class__.local_arraytype):
                # assign local array to self (distributed array)
                window = [slice(None)] * len(self.shape)
                subarrays = []
                for i in range(len(self.target_offsets)):
                    begin = self.target_offsets[i]
                    end = self.target_offsets[i+1] if i+1 < len(self.target_offsets) else self.shape[self.distaxis]
                    window[self.distaxis] = slice(begin, end)
                    subarrays.append(value[window])
                self.view.scatter("subarray", subarrays, targets=self.target_ranks)
                self.view.execute("%s[:] = subarray[0]" % self.name, targets=self.target_ranks)
                return

            other = value.dist_like(self)
            self.view.execute("%s[:] = %s" % (self.name, other.name), targets=self.target_ranks)
            return

        if not isinstance(key, list) and not isinstance(key, tuple):
            key = (key,)

        assert len(key) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                 % (len(key), len(self.shape))

        # value assignment (key == list of indices)
        if all(type(k) is int for k in key):
            dist_idx = key[self.distaxis]
            target_idx = np.searchsorted(self.target_offsets, dist_idx, side="right") - 1
            local_idx = list(key)
            local_idx[self.distaxis] = dist_idx - self.target_offsets[target_idx]
            self.view.push({'value' : value}, targets=self.target_ranks[target_idx])
            self.view.execute("%s%s = value" % (self.name, repr(local_idx)), targets=self.target_ranks[target_idx])
            return

        # assign value to sub-array of self
        sub_array = self[key]
        sub_array[:] = value

    def __str__(self):
        return self.gather().__str__()

    def __repr__(self):
        return self.name

    def gather(self):
        """Gathers local instances of {local_arraytype_name} from *engines*, concatenates them and returns
        the result.

        :return: instance of {local_arraytype_name}
        """
        local_arrays = self.view.pull(self.name, targets=self.target_ranks)

        result = self.__class__.local_arraytype(shape=self.shape, dtype=self.dtype, **self.kwargs)
        window = [slice(None)] * len(self.shape)
        for i, local_array in zip(range(len(self.target_offsets)), local_arrays):
            begin = self.target_offsets[i]
            end = self.target_offsets[i+1] if i+1 < len(self.target_offsets) else self.shape[self.distaxis]
            window[self.distaxis] = slice(begin, end)
            result[window] = local_array

        return result

    def copy(self):
        """Returns a hard copy of this array.
        """
        assert self.__class__.may_allocate == True, "{0} is not allowed to allocate new memory.".format(self.__class__.__name__)

        result = self.__class__(self.shape, self.dtype, self.distaxis, self.target_offsets, self.target_ranks, no_allocation=True, **self.kwargs)
        self.view.execute("%s = %s.copy()" % (result.name, self.name), targets=self.target_ranks)
        return result

    def dist_like(self, other):
        """Redistributes a copy of this array (*self*) like *other* and returns the result.
        Checks whether redistribution is necessary and returns *self* if not.

        Redistribution involves inter-engine communication.

        :param other: target array
        :type other: distributed array
        :raises AssertionError: if the shapes of *self* and *other* don't match.
        :raises AssertionError: if *self* and *other* are distributed along distinct axes.
        :return: new array with the same content as *self* but distributed like *other*.
            If *self* is already distributed like *other* nothing is done and *self* is returned.
        """
        assert self.shape == other.shape,\
            "Shapes do not match: " + str(self.shape) + " <-> " + str(other.shape)
        assert self.distaxis == other.distaxis # todo: add this feature

        # if self is already distributed like *other* do nothing
        if np.array_equal(self.target_offsets, other.target_offsets):
            if self.target_ranks == other.target_ranks:
                return self

        assert self.__class__.may_allocate, "{0} is not allowed to allocate new memory.".format(self.__class__.__name__)

        # communication meta-data
        src_targets = [[]]
        src_tags = [[]]
        src_distaxis_sizes = [[]]
        dest_targets = [[]]
        dest_tags = [[]]
        dest_distaxis_sizes = [[]]

        self_idx = 0
        other_idx = 0
        tag = 0
        begin = 0
        # loop the common decomposition of self and other.
        # the current partition is given by [begin, end)
        while not begin == self.shape[self.distaxis]:
            end_self = self.target_offsets[self_idx+1] if self_idx+1 < len(self.target_offsets) else self.shape[self.distaxis]
            end_other = other.target_offsets[other_idx+1] if other_idx+1 < len(other.target_offsets) else other.shape[other.distaxis]
            end = min(end_self, end_other)
            partition_size = end - begin

            # source's meta-data
            src_targets[self_idx].append(other.target_ranks[other_idx])
            src_tags[self_idx].append(tag)
            src_distaxis_sizes[self_idx].append(partition_size)

            # destination's meta-data
            dest_targets[other_idx].append(self.target_ranks[self_idx])
            dest_tags[other_idx].append(tag)
            dest_distaxis_sizes[other_idx].append(partition_size)

            # go to the next common partition
            if end == end_self:
                if not end == self.shape[self.distaxis]:
                    src_targets.append([])
                    src_tags.append([])
                    src_distaxis_sizes.append([])
                self_idx += 1
            if end == end_other:
                if not end == self.shape[self.distaxis]:
                    dest_targets.append([])
                    dest_tags.append([])
                    dest_distaxis_sizes.append([])
                other_idx += 1

            begin = end
            tag += 1

        # push communication meta-data to engines
        self.view.scatter('src_targets', src_targets, targets=self.target_ranks)
        self.view.scatter('src_tags', src_tags, targets=self.target_ranks)
        self.view.scatter('src_distaxis_sizes', src_distaxis_sizes, targets=self.target_ranks)

        self.view.scatter('dest_targets', dest_targets, targets=other.target_ranks)
        self.view.scatter('dest_tags', dest_tags, targets=other.target_ranks)
        self.view.scatter('dest_distaxis_sizes', dest_distaxis_sizes, targets=other.target_ranks)

        # result ndarray
        result = self.__class__(self.shape, self.dtype, self.distaxis, other.target_offsets, other.target_ranks, False, **self.kwargs)

        self.__class__.interengine_copier(self, result)

        return result

    def __elementwise_op__(self, op, *args):
        args = [arg.dist_like(self) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)

        result = self.__class__(self.shape, self.dtype, self.distaxis, self.target_offsets, self.target_ranks, no_allocation=True, **self.kwargs)

        self.view.execute("{0} = {1}.{2}({3}); dtype={0}.dtype".format(repr(result), repr(self), op, arg_string), targets=self.target_ranks)
        result.dtype = self.view.pull("dtype", targets=result.target_ranks[0])
        result.nbytes = np.dtype(result.dtype).itemsize * np.prod(result.shape)
        return result

    def __elementwise_iop__(self, op, *args):
        args = [arg.dist_like(self) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)
        self.view.execute("%s = %s.%s(%s)" % (repr(self), repr(self), op, arg_string), targets=self.target_ranks)
        return self

#----------------------------------------------------------------

import types

def distribute(local_arraytype, newclassname, target_modulename, interengine_copier=None, may_allocate = True):
    binary_ops = ["add", "sub", "mul", "floordiv", "div", "mod", "pow", "lshift", "rshift", "and", "xor", "or"]

    binary_iops = ["__i" + op + "__" for op in binary_ops]
    binary_rops = ["__r" + op + "__" for op in binary_ops]
    binary_ops = ["__" + op + "__" for op in binary_ops]
    unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__long__", "__float__", "__oct__", "__hex__"]
    comp_ops = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]

    special_ops_avail = set(name for name in local_arraytype.__dict__.keys() if name.endswith("__"))

    make_special_op = lambda op: lambda self, *args: self.__elementwise_op__(op, *args)
    make_special_iop = lambda op: lambda self, *args: self.__elementwise_iop__(op, *args)

    special_ops_dict = {op : make_special_op(op) for op in \
        set(binary_ops + binary_rops + unary_ops + comp_ops) & special_ops_avail}
    special_iops_dict = {op : make_special_iop(op) for op in set(binary_iops) & special_ops_avail}

    formated_doc_funs = ("__init__", "gather")

    result_dict = dict(DistributedGenericArray.__dict__)

    # docs
    result_dict["__doc__"] = result_dict["__doc__"].format(\
        local_arraytype_name=local_arraytype.__module__ + "." +  local_arraytype.__name__,
        arraytype_name=newclassname)
    # copy methods which have formated docstrings because their docstrings are going to be modified
    copied_methods = {k : types.FunctionType(v.func_code, v.func_globals, name=v.func_name, argdefs=v.func_defaults)\
        for k,v in result_dict.items() if k in formated_doc_funs}
    result_dict.update(copied_methods)

    result_dict.update(special_ops_dict)
    result_dict.update(special_iops_dict)

    result = type(newclassname, (), result_dict)
    result.local_arraytype = local_arraytype
    result.target_modulename = target_modulename
    result.interengine_copier = interengine_copier
    result.may_allocate = may_allocate

    # docs
    for method in (v for k,v in result.__dict__.items() if k in formated_doc_funs):
        method.__doc__ = method.__doc__.format(\
            local_arraytype_name=local_arraytype.__module__ + "." +  local_arraytype.__name__,
            arraytype_name=newclassname)


    return result

def generate_ufuncs(ufunc_names, target_modulename):

    def ufunc_wrapper(ufunc_name, args, kwargs):
        arg0 = args[0]
        args = [arg.dist_like(arg0) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)

        view = com.getView()
        result = arg0.__class__(arg0.shape, arg0.dtype, arg0.distaxis, arg0.target_offsets, arg0.target_ranks, no_allocation=True, **arg0.kwargs)

        view.execute("{0} = {1}({2}); dtype={0}.dtype".format(repr(result), ufunc_name, arg_string), targets=arg0.target_ranks)
        result.dtype = view.pull("dtype", targets=result.target_ranks[0])
        result.nbytes = np.dtype(result.dtype).itemsize * np.prod(result.shape)
        return result

    make_ufunc = lambda ufunc_name: lambda *args, **kwargs: ufunc_wrapper(target_modulename + "." + ufunc_name, args, kwargs)

    return {ufunc_name: make_ufunc(ufunc_name) for ufunc_name in ufunc_names}

def generate_factories(arraytype, factory_names, dtype_default):

    def factory_wrapper(factory_name, shape, dtype, distaxis, kwargs):
        result = arraytype(shape, dtype, distaxis, None, None, True, **kwargs)

        localshapes = []
        for i in range(len(result.target_offsets)-1):
            localshape = np.array(result.shape)
            localshape[distaxis] = result.target_offsets[i+1] - result.target_offsets[i]
            localshapes.append(localshape)
        localshape = np.array(result.shape)
        localshape[distaxis] = result.shape[distaxis] - result.target_offsets[-1]
        localshapes.append(localshape)

        view = com.getView()
        view.scatter('localshape', localshapes, targets=result.target_ranks)
        view.push({'kwargs' : kwargs, 'dtype' : dtype}, targets=result.target_ranks)

        view.execute("{0} = {1}(shape=localshape[0], dtype=dtype, **kwargs)".format(result.name, factory_name),\
            targets=result.target_ranks)
        return result

    make_factory = lambda factory_name: lambda shape, dtype=dtype_default, distaxis=0, **kwargs:\
        factory_wrapper(arraytype.target_modulename + "." + factory_name, shape, dtype, distaxis, kwargs)

    factories_dict = {factory_name : make_factory(factory_name) for factory_name in factory_names}

    # add docstrings
    for name, factory in factories_dict.items():
        factory.__name__ = name
        factory.__doc__ = \
        """Create a *{0}* instance. This function calls its local counterpart *{1}* on each :term:`engine`.

        :param ints shape: shape of array
        :param dtype: datatype of a single element
        :param int distaxis: distributed axis
        :param kwargs: keyword arguments are passed to the local function *{1}*
        """.format(arraytype.__name__, str(arraytype.local_arraytype.__module__) + "." + name)

    return factories_dict

def generate_factories_like(arraytype, factory_names):

    def factory_like_wrapper(factory_name, other, kwargs):
        result = arraytype(other.shape, other.dtype, other.distaxis, other.target_offsets, other.target_ranks, True, **kwargs)
        view = com.getView()
        view.push({'kwargs' : kwargs}, targets=result.target_ranks)
        view.execute("{0} = {1}({2}, **kwargs)".format(result.name, factory_name, other.name), targets=result.target_ranks)
        return result

    make_factory = lambda factory_name: lambda other, **kwargs: \
        factory_like_wrapper(arraytype.target_modulename + "." + factory_name, other, kwargs)

    factories_dict = {factory_name : make_factory(factory_name) for factory_name in factory_names}

    # add docstrings
    for name, factory in factories_dict.items():
        factory.__name__ = name
        factory.__doc__ = \
        """Create a *{0}* instance with the same shape, dtype and distribution as ``other``.
        This function calls its local counterpart *{1}* on each :term:`engine`.

        :param other: other array
        :param kwargs: keyword arguments are passed to the local function *{1}*
        """.format(arraytype.__name__, str(arraytype.local_arraytype.__module__) + "." + name)

    return factories_dict