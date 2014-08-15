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
"""The *arrayOfStructs* module addresses the common problem when dealing with
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

    fields = pyDive.arrayOfStructs(treeOfArrays)

    half = fields[::2]["FieldE/x"]
    # equivalent to
    half = fields["FieldE/x"][::2]
    # equivalent to
    half = fields["FieldE"]["x"][::2]
    # equivalent to
    half = fields["FieldE"][::2]["x"]

The example shows that in fact *fields* can be treated as an array-of-structures
**or** a structure-of-arrays depending on what is more appropriate.

The goal is to make the virtual *array-of-structs*-object look like a real array even ready for passing
to foreign functions that expect a "real" array which at least inherits e.g. from a numpy-array.
Therefore the virtual *array-of-structs*-object inherits from the class type of its arrays.
This makes it for instance compatible to :mod:`pyDive.algorithm`.
"""

import sys
import os
# check whether this code is executed on target or not
onTarget = os.environ.get("onTarget", 'False')
if onTarget == 'False':
    import IPParallelClient as com
    from ndarray.ndarray import ndarray as ndarray
    from h5_ndarray.h5_ndarray import h5_ndarray as h5_ndarray
    from IPython.parallel import interactive
    import debug
import numpy as np
from abc import ABCMeta

def makeTree_like(tree, expression):
    def traverseTree(outTree, inTree):
        for key, value in inTree.items():
            if type(value) is dict:
                outTree[key] = {}
                traverseTree(outTree[key], inTree[key])
            else:
                outTree[key] = expression(value)
    outTree = {}
    traverseTree(outTree, tree)
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

arrayOfStructs_id = 0

class arrayOfStructsClass(object):

    def __init__(self, structOfArrays):
        items = [item for item in treeItems(structOfArrays)]
        self.firstArray = items[0][1]
        assert all(type(a) == type(self.firstArray) for name, a in items),\
            "all arrays in 'structOfArrays' must be of the same type"
        assert all(a.shape == self.firstArray.shape for name, a in items),\
            "all arrays in 'structOfArrays' must have the same shape"

        self.shape = self.firstArray.shape
        self.dtype = makeTree_like(structOfArrays, lambda a: a.dtype)
        self.nbytes = sum(a.nbytes for name, a in items)
        self.structOfArrays = structOfArrays

        if onTarget == 'False' and isinstance(self, ndarray):
            #assert all(a.targets_in_use == firstArray.targets_in_use for name, a in items),\
            #    "all ndarrays in structure-of-arrays ('structOfArrays') must have an identical 'targets_in_use' attribute"

            self.distaxis = self.firstArray.distaxis
            self.idx_ranges = self.firstArray.idx_ranges
            self.targets_in_use = self.firstArray.targets_in_use
            view = com.getView()
            self.view = view

            # generate a unique variable name used on target representing this instance
            global arrayOfStructs_id
            self.name = 'arrayOfStructsObj' + str(arrayOfStructs_id)
            arrayOfStructs_id += 1

            # create an arrayOfStructsClass object consisting of the numpy arrays on the targets in use
            names_tree = makeTree_like(structOfArrays, lambda a: repr(a))

            view.push({'names_tree' : names_tree}, targets=self.targets_in_use)

            ar = view.execute('''\
                structOfArrays = arrayOfStructs.makeTree_like(names_tree, lambda a_name: globals()[a_name])
                %s = arrayOfStructs.arrayOfStructs(structOfArrays)''' % self.name,\
                targets=self.targets_in_use,block=False)

            debug.wait_watching_stdout(ar)

        if onTarget == 'False' and isinstance(self, h5_ndarray):
            self.distaxis = self.firstArray.distaxis

    def __del__(self):
        if onTarget == 'False' and isinstance(self, ndarray):
            # delete remote arrayOfStructs object
            self.view.execute('del %s' % self.name, targets=self.targets_in_use)

    def __repr__(self):
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

        result = "CustomStructOfArrays<type: " + str(type(self.firstArray)) +\
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
                return arrayOfStructs(node)
            else:
                # leaf
                return node

        # slicing
        # -------
        if not isinstance(args, list) and not isinstance(args, tuple):
            args = (args,)

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                % (len(args), len(self.shape))

        result = makeTree_like(self.structOfArrays, lambda a: a[args])

        # if args is a list of indices then return a single data value tree
        if all(type(arg) is int for arg in args):
            return result

        return arrayOfStructs(result)

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
        if not isinstance(args, list) and not isinstance(args, tuple):
            args = (args,)

        assert len(args) == len(self.shape),\
            "number of arguments (%d) does not correspond to the dimension (%d)"\
                % (len(args), len(self.shape))

        def doArrayAssignmentWithSlice(treeA, treeB, name, arrayA, arrayB):
            treeA[name][args] = arrayB

        visitTwoTrees(self.structOfArrays, other.structOfArrays, doArrayAssignmentWithSlice)

def arrayOfStructs(structOfArrays):
    """Convert a *structure-of-arrays* into a virtual *array-of-structures*.

    :param structOfArrays: tree-like dictionary of arrays.
    :raises AssertionError: if the *arrays-types* do not match. Datatypes may differ.
    :raises AssertionError: if the shapes do not match.
    :return: Custom object representing a virtual array whose elements have the same tree-like structure
        as *structOfArrays*. It inherits from *array-type*.
    """
    first_item = treeItems(structOfArrays).next()
    array_type = type(first_item[1])

    class CustomArrayOfStructs(arrayOfStructsClass, array_type):
        __metaclass__ = type

        def __new__(cls, *args):
            return cls.__bases__[1].__new__(cls, shape=[0])

        def __init__(self, structOfArrays):
            super(CustomArrayOfStructs, self).__init__(structOfArrays)

        def __setattr__(self, name, value):
            self.__dict__[name] = value

        def __getattribute__(self, name):
            try:
                return object.__getattribute__(self, "__dict__")[name]
            except KeyError:
                return CustomArrayOfStructs.__bases__[1].__getattribute__(self, name)

    return CustomArrayOfStructs(structOfArrays)
