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

__doc__ =\
"""The *structured* module addresses the common problem when dealing with
structured data: While the user likes an array-of-structures layout the machine prefers a structure-of-arrays.
In pyDive the method of choice is a *virtual* *array-of-structures*-object. It holds array-like attributes
such as shape and dtype and allows for slicing but is operating on a structure-of-arrays internally.

Example: ::

    ...
    treeOfArrays = {"FieldE" :
                        {"x" : fielde_x,
                         "y" : fielde_y,
                         "z" : fielde_z},
                    "FieldB" :
                        {"x" : fieldb_x,
                         "y" : fieldb_y,
                         "z" : fieldb_z}
                    }

    fields = pyDive.structured(treeOfArrays)

    half = fields[::2]["FieldE/x"]
    # equivalent to
    half = fields["FieldE/x"][::2]
    # equivalent to
    half = fields["FieldE"]["x"][::2]
    # equivalent to
    half = fields["FieldE"][::2]["x"]

    # equivalent to
    half = fields.FieldE.x[::2]

The example shows that in fact *fields* can be treated as an array-of-structures
**or** a structure-of-arrays depending on what is more appropriate.

The goal is to make the virtual *array-of-structs*-object look like a real array. Therefore
every method call or operation is forwarded to the individual arrays.::

    new_field = fields.FieldE.astype(np.int) + fields.FieldB.astype(np.float)

Here the forwarded method calls are ``astype`` and ``__add__``.
"""

import sys
import os
# check whether this code is executed on target or not
onTarget = os.environ.get("onTarget", 'False')
if onTarget == 'False':
    import IPParallelClient as com
    from IPython.parallel import interactive
import numpy as np

def makeTree_fromTree(tree, expression):
    def traverseTree(outTree, inTree):
        for key, value in inTree.items():
            if type(value) is dict:
                outTree[key] = {}
                traverseTree(outTree[key], value)
            else:
                outTree[key] = expression(value)
    outTree = {}
    traverseTree(outTree, tree)
    return outTree

def makeTree_fromTwoTrees(treeA, treeB, expression):
    def traverseTrees(outTree, inTreeA, inTreeB):
        for key, valueA in inTreeA.items():
            valueB = inTreeB[key]
            if type(valueA) is dict:
                outTree[key] = {}
                traverseTrees(outTree[key], valueA, valueB)
            else:
                outTree[key] = expression(valueA, valueB)
    outTree = {}
    traverseTrees(outTree, treeA, treeB)
    return outTree

def visitTwoTrees(treeA, treeB, visitor):
    # traverse trees
    for key, valueA in treeA.items():
        valueB = treeB[key]
        if type(valueA) is dict:
            visitTwoTrees(valueA, valueB, visitor)
        else:
            visitor(treeA, treeB, key, valueA, valueB)
    return treeA, treeB

# generator object which iterates the tree's leafs
def treeItems(tree):
    items = list(tree.items())
    while items:
        key, value = items.pop()
        if type(value) is dict:
            items += list(value.items())
        else:
            yield key, value

class ForeachLeafCall(object):
    def __init__(self, tree, op):
        self.tree = tree
        self.op = op

    def __call__(self, *args, **kwargs):
        def apply_unary(a):
            f = getattr(a, self.op)
            return f(*args, **kwargs)
        def apply_binary(a, b):
            f = getattr(a, self.op)
            return f(b, *args[1:], **kwargs)

        if args and type(args[0]).__name__ == "VirtualArrayOfStructs":
            structOfArrays = makeTree_fromTwoTrees(self.tree, args[0].structOfArrays, apply_binary)
        else:
            structOfArrays = makeTree_fromTree(self.tree, apply_unary)
        return structured(structOfArrays)


arrayOfStructs_id = 0

class VirtualArrayOfStructs(object):
    def __init__(self, structOfArrays):
        items = [item for item in treeItems(structOfArrays)]
        self.firstArray = items[0][1]
        self.arraytype = type(self.firstArray)
        assert all(type(a) == self.arraytype for name, a in items),\
            "all arrays in 'structOfArrays' must be of the same type: " +\
            str({name : type(a) for name, a in items})
        assert all(a.shape == self.firstArray.shape for name, a in items),\
            "all arrays in 'structOfArrays' must have the same shape: " +\
            str({name : a.shape for name, a in items})

        self.shape = self.firstArray.shape
        self.dtype = makeTree_fromTree(structOfArrays, lambda a: a.dtype)
        self.nbytes = sum(a.nbytes for name, a in items)
        self.structOfArrays = structOfArrays
        self.has_local_instance = False

    def __del__(self):
        if onTarget == 'False' and self.has_local_instance:
            # delete remote structured object
            self.view.execute('del %s' % self.name, targets=self.target_ranks)

    def __getattr__(self, name):
        if hasattr(self.firstArray, name):
            assert hasattr(getattr(self.firstArray, name), "__call__"),\
                "Unlike method access, attribute access of individual arrays is not supported."
            return ForeachLeafCall(self.structOfArrays, name)

        return self[name]

    def __special_operation__(self, op, *args):
        return ForeachLeafCall(self.structOfArrays, op)(*args)

    def __repr__(self):
        # if arrays are distributed create a local representation of this object on engine
        if onTarget == 'False' and not self.has_local_instance and hasattr(self.firstArray, "target_ranks"):
            items = [item for item in treeItems(self.structOfArrays)]
            assert all(self.firstArray.is_distributed_like(a) for name, a in items),\
                "Cannot create a local virtual array-of-structs because not all arrays are distributed equally."

            self.distaxes = self.firstArray.distaxes
            self.target_offsets = self.firstArray.target_offsets
            self.target_ranks = self.firstArray.target_ranks
            view = com.getView()
            self.view = view

            # generate a unique variable name used on target representing this instance
            global arrayOfStructs_id
            self.name = 'arrayOfStructsObj' + str(arrayOfStructs_id)
            arrayOfStructs_id += 1

            # create a VirtualArrayOfStructs object containing the local arrays on the targets in use
            names_tree = makeTree_fromTree(self.structOfArrays, lambda a: repr(a))

            view.push({'names_tree' : names_tree}, targets=self.target_ranks)

            view.execute('''\
                structOfArrays = structured.makeTree_fromTree(names_tree, lambda a_name: globals()[a_name])
                %s = structured.structured(structOfArrays)''' % self.name,\
                targets=self.target_ranks)

            self.has_local_instance = True

        return self.name

    def __str__(self):
        def printTree(tree, indent, result):
            for key, value in tree.items():
                if type(value) is dict:
                    result += indent + key + ":\n"
                    result = printTree(tree[key], indent + "  ", result)
                else:
                    result += indent + key + " -> " + str(value.dtype) + "\n"
            return result

        result = "VirtualArrayOfStructs<array-type: " + str(type(self.firstArray)) +\
            ", shape: " + str(self.shape) + ">:\n"
        return printTree(self.structOfArrays, "  ", result)

    def __getitem__(self, args):
        # component access
        # ----------------
        if type(args) is str:
            node = self.structOfArrays # root node
            path = args.split('/')
            for node_name in path:
                node = node[node_name]
            if type(node) is dict:
                # node
                return structured(node)
            else:
                # leaf
                return node

        # slicing
        # -------
        result = makeTree_fromTree(self.structOfArrays, lambda a: a[args])

        # if args is a list of indices then return a single data value tree
        if type(args) not in (list, tuple):
            args = (args,)
        if all(type(arg) is int for arg in args):
            return result

        return structured(result)

    def __setitem__(self, args, other):
        # component access
        # ----------------
        if type(args) is str:
            node = self.structOfArrays # root node
            path = args.split('/')
            for node_name in path[:-1]:
                node = node[node_name]
            last_node_name = path[-1]
            if type(node[last_node_name]) is dict:
                # node
                def doArrayAssignment(treeA, treeB, name, arrayA, arrayB):
                    treeA[name] = arrayB

                visitTwoTrees(node[last_node_name], other.structOfArrays, doArrayAssignment)
            else:
                # leaf
                node[last_node_name] = other
            return

        # slicing
        # -------
        def doArrayAssignmentWithSlice(treeA, treeB, name, arrayA, arrayB):
            arrayA[args] = arrayB

        visitTwoTrees(self.structOfArrays, other.structOfArrays, doArrayAssignmentWithSlice)

# Add special methods like "__add__", "__sub__", ... that call __special_operation__
# forwarding them to the individual arrays.
# All ordinary methods are forwarded by __getattr__

binary_ops = ["add", "sub", "mul", "floordiv", "div", "mod", "pow", "lshift", "rshift", "and", "xor", "or"]

binary_iops = ["__i" + op + "__" for op in binary_ops]
binary_rops = ["__r" + op + "__" for op in binary_ops]
binary_ops = ["__" + op + "__" for op in binary_ops]
unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__long__", "__float__", "__oct__", "__hex__"]
comp_ops = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]

make_special_op = lambda op: lambda self, *args: self.__special_operation__(op, *args)

special_ops_dict = {op : make_special_op(op) for op in binary_ops + binary_rops + unary_ops + comp_ops}

from types import MethodType

for name, func in special_ops_dict.items():
    setattr(VirtualArrayOfStructs, name, MethodType(func, None, VirtualArrayOfStructs))


def structured(structOfArrays):
    """Convert a *structure-of-arrays* into a virtual *array-of-structures*.

    :param structOfArrays: tree-like (dict-of-dicts) dictionary of arrays.
    :raises AssertionError: if the *arrays-types* do not match. Datatypes may differ.
    :raises AssertionError: if the shapes do not match.
    :return: Custom object representing a virtual array whose elements have the same tree-like structure
        as *structOfArrays*.
    """

    return VirtualArrayOfStructs(structOfArrays)
