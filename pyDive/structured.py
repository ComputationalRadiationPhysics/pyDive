# Copyright 2014-2016 Heiko Burau
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

import os
import operator
# check whether this code is executed on target or not
onTarget = os.environ.get("onTarget", 'False')
if onTarget == 'False':
    from . import ipyParallelClient as com

arrayOfStructs_id = 0

__doc__ = \
    """The *structured* module addresses the common problem when dealing with
    structured data: While the user likes an array-of-structures layout the machine
    prefers a structure-of-arrays. In pyDive the method of choice is a
    *virtual* *array-of-structures*-object. It holds array-like attributes such as
    shape and dtype and allows for slicing but is operating on a structure-of-arrays internally.

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


class VirtualArrayOfStructs(object):
    def __init__(self, structOfArrays):
        self.structOfArrays = structOfArrays

        items = list(self.items())
        self.firstArray = items[0][1]
        self.arraytype = type(self.firstArray)
        assert all(type(a) == self.arraytype for name, a in items),\
            "all arrays in 'structOfArrays' must be of the same type: " +\
            str({name: type(a) for name, a in items})
        assert all(a.shape == self.firstArray.shape for name, a in items),\
            "all arrays in 'structOfArrays' must have the same shape: " +\
            str({name: a.shape for name, a in items})

        self.shape = self.firstArray.shape
        self.dtype = self.map(lambda a: a.dtype)
        self.nbytes = sum(a.nbytes for name, a in items)
        self.has_local_instance = False

    def __del__(self):
        if onTarget == 'False' and self.has_local_instance:
            # delete remote structured object
            self.view.execute('del %s' % self.name, targets=self.decomposition.ranks)

    def __getattr__(self, name):
        if hasattr(self.firstArray, name):
            attr_tree = self.map(lambda a: getattr(a, name))

            if not callable(getattr(self.firstArray, name)):
                return attr_tree

            def foreachLeafCall(*args, **kwargs):
                aos = tuple(arg.structOfArrays for arg in args
                            if type(arg).__name__ is "VirtualArrayOfStructs")
                misc_args = tuple(arg for arg in args if type(arg) is not VirtualArrayOfStructs)
                return structured(
                    map_trees(lambda method, *a: method(*(a + misc_args), **kwargs),
                              *((attr_tree,) + aos)))

            return foreachLeafCall

        return self[name]

    def __special_operation__(self, op, *args):
        if args:
            if type(args[0]) is VirtualArrayOfStructs:
                return structured(
                    map_trees(lambda a, b: getattr(a, op)(b),
                              self.structOfArrays, args[0].structOfArrays))
            else:
                return structured(self.map(lambda a: getattr(a, op)(args[0])))
        else:
            return structured(self.map(lambda a: getattr(a, op)()))

    def __repr__(self):
        # if arrays are distributed create a local representation of this object on engine
        if onTarget == 'False' and not self.has_local_instance and hasattr(self.firstArray, "decomposition"):
            assert all(self.firstArray.is_distributed_like(a) for name, a in self.items()),\
                """Cannot create a local virtual array-of-structs because
                not all arrays are distributed equally."""

            self.distaxes = self.firstArray.distaxes
            self.decomposition = self.firstArray.decomposition
            view = com.getView()
            self.view = view

            # generate a unique variable name used on target representing this instance
            global arrayOfStructs_id
            self.name = 'arrayOfStructsObj' + str(arrayOfStructs_id)
            arrayOfStructs_id += 1

            # create a VirtualArrayOfStructs object containing the local arrays on the targets in use
            names_tree = self.map(repr)

            view.push({'names_tree': names_tree}, targets=self.decomposition.ranks)

            view.execute('''\
                         structOfArrays = structured.map_trees(globals().get, names_tree)
                         %s = structured.structured(structOfArrays)''' % self.name,
                         targets=self.decomposition.ranks)

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

    def __iter__(self):
        paths = list(self.structOfArrays)
        values = list(self.structOfArrays.values())
        while paths:
            path = paths.pop()
            value = values.pop()
            if type(value) is dict:
                paths += [path + "/" + k for k in value]
                values += list(value.values())
            else:
                yield path

    def items(self):
        items = list(self.structOfArrays.items())
        while items:
            key, value = items.pop()
            if type(value) is dict:
                items += list(value.items())
            else:
                yield key, value

    def map(self, f):
        def mapTree(tree):
            if type(tree) is not dict:
                return f(tree)
            return {k: mapTree(v) for k, v in tree.items()}

        return mapTree(self.structOfArrays)

    def __getitem__(self, args):
        # component access
        # ----------------
        if type(args) is str:
            node = self.structOfArrays  # root node
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
        result = self.map(lambda a: a[args])

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
            node = self.structOfArrays  # root node
            path = args.split('/')
            for node_name in path[:-1]:
                node = node[node_name]
            last_node_name = path[-1]
            if type(node[last_node_name]) is dict:
                # node
                node[last_node_name] = other.structOfArrays
            else:
                # leaf
                node[last_node_name] = other
            return

        # slicing
        # -------
        map_trees(lambda a, b: operator.setitem(a, args, b), self.structOfArrays, other.structOfArrays)


# Add special methods like "__add__", "__sub__", ... that call __special_operation__
# forwarding them to the individual arrays.
# All ordinary methods are forwarded by __getattr__

binary_ops = ["add", "sub", "mul", "floordiv", "div", "mod", "pow",
              "lshift", "rshift", "and", "xor", "or"]

binary_iops = ["__i" + op + "__" for op in binary_ops]
binary_rops = ["__r" + op + "__" for op in binary_ops]
binary_ops = ["__" + op + "__" for op in binary_ops]
unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__", "__complex__",
             "__int__", "__long__", "__float__", "__oct__", "__hex__"]
comp_ops = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]

make_special_op = lambda op: lambda self, *args: self.__special_operation__(op, *args)

special_ops_dict = {op: make_special_op(op) for op in
                    binary_ops + binary_rops + unary_ops + comp_ops}

# add everything to class
for name, func in special_ops_dict.items():
    setattr(VirtualArrayOfStructs, name, func)

# ---------------------------------------------------------------------------------------------


def map_trees(f, *trees):
    if type(trees[0]) is not dict:
        return f(*trees)
    return {k: map_trees(f, *[t[k] for t in trees]) for k in trees[0]}


def flat_values(tree):
    values = list(tree.values())
    while values:
        value = values.pop()
        if type(value) is dict:
            values += list(value.values())
        else:
            yield value


def structured(data):
    """Create a new virtual *array-of-structures*.

    :param data: input data (see below)
    :raises AssertionError: if the *arrays-types* do not match. Datatypes may differ.
    :raises AssertionError: if the shapes do not match.
    :return: Custom object representing a virtual array whose elements have the same tree-like structure
        as `data`.

    `data` can either be a tree-like (dict-of-dicts) dictionary of arrays
    or a list of key/value pairs. Nodes a separated by `/`. Example: ::

        fields = structured([("fieldE/x", fieldE_x), ("fieldE/y", fieldE_y), ("fieldB/z", fieldB_z)])
        # or:
        fields = structured({"fieldE" : {"x" : fieldE_x, "y" : fieldE_y}, "fieldB" : {"z" : fieldB_z} })
    """

    if type(data) is dict:
        return VirtualArrayOfStructs(data)

    assert type(data) in (tuple, list),\
        "`data` must be either a dictionary, a list or a tuple."

    def update_tree(tree, nodes, array):
        if len(nodes) == 1:
            tree[nodes[0]] = array
            return
        if not nodes[0] in tree:
            tree[nodes[0]] = {}
        update_tree(tree[nodes[0]], nodes[1:], array)

    structOfArrays = {}
    for path, array in data:
        nodes = path.strip("/").split("/")
        update_tree(structOfArrays, nodes, array)

    return VirtualArrayOfStructs(structOfArrays)
