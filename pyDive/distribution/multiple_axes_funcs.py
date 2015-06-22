# -*- coding: utf-8 -*-

import types
from multiple_axes import DistributedGenericArray
import numpy as np
import pyDive.IPParallelClient as com

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

def generate_factories(arraytype, factory_names, dtype_default):

    def factory_wrapper(factory_name, shape, dtype, distaxes, kwargs):
        result = arraytype(shape, dtype, distaxes, None, None, True, **kwargs)

        target_shapes = result.target_shapes()

        view = com.getView()
        view.scatter('target_shape', target_shapes, targets=result.target_ranks)
        view.push({'kwargs' : kwargs, 'dtype' : dtype}, targets=result.target_ranks)

        view.execute("{0} = {1}(shape=target_shape[0], dtype=dtype, **kwargs)".format(result.name, factory_name),\
            targets=result.target_ranks)
        return result

    make_factory = lambda factory_name: lambda shape, dtype=dtype_default, distaxes='all', **kwargs:\
        factory_wrapper(arraytype.target_modulename + "." + factory_name, shape, dtype, distaxes, kwargs)

    factories_dict = {factory_name : make_factory(factory_name) for factory_name in factory_names}

    # add docstrings
    for name, factory in factories_dict.items():
        factory.__name__ = name
        factory.__doc__ = \
        """Create a *{0}* instance. This function calls its local counterpart *{1}* on each :term:`engine`.

        :param ints shape: shape of array
        :param dtype: datatype of a single element
        :param ints distaxes: distributed axes
        :param kwargs: keyword arguments are passed to the local function *{1}*
        """.format(arraytype.__name__, str(arraytype.local_arraytype.__module__) + "." + name)

    return factories_dict

def generate_factories_like(arraytype, factory_names):

    def factory_like_wrapper(factory_name, other, kwargs):
        result = arraytype(other.shape, other.dtype, other.distaxes, other.target_offsets, other.target_ranks, True, **kwargs)
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

def generate_ufuncs(ufunc_names, target_modulename):

    def ufunc_wrapper(ufunc_name, args, kwargs):
        arg0 = args[0]
        args = [arg.dist_like(arg0) if hasattr(arg, "target_ranks") else arg for arg in args]
        arg_names = [repr(arg) for arg in args]
        arg_string = ",".join(arg_names)

        view = com.getView()
        result = arg0.__class__(arg0.shape, arg0.dtype, arg0.distaxes, arg0.target_offsets, arg0.target_ranks, no_allocation=True, **arg0.kwargs)

        view.execute("{0} = {1}({2}); dtype={0}.dtype".format(repr(result), ufunc_name, arg_string), targets=arg0.target_ranks)
        result.dtype = view.pull("dtype", targets=result.target_ranks[0])
        result.nbytes = np.dtype(result.dtype).itemsize * np.prod(result.shape)
        return result

    make_ufunc = lambda ufunc_name: lambda *args, **kwargs: ufunc_wrapper(target_modulename + "." + ufunc_name, args, kwargs)

    return {ufunc_name: make_ufunc(ufunc_name) for ufunc_name in ufunc_names}