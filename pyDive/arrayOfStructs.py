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
        firstArray = items[0][1]
        assert all(type(a) == type(firstArray) for name, a in items),\
            "all arrays in 'structOfArrays' must be of the same type"
        assert all(a.shape == firstArray.shape for name, a in items),\
            "all arrays in 'structOfArrays' must have the same shape"

        self.shape = firstArray.shape
        self.dtype = makeTree_like(structOfArrays, lambda a: a.dtype)
        self.nbytes = sum(a.nbytes for name, a in items)
        self.structOfArrays = structOfArrays

        if onTarget == 'False' and isinstance(self, ndarray):
            #assert all(a.targets_in_use == firstArray.targets_in_use for name, a in items),\
            #    "all ndarrays in structure-of-arrays ('structOfArrays') must have an identical 'targets_in_use' attribute"

            self.distaxis = firstArray.distaxis
            self.idx_ranges = firstArray.idx_ranges
            self.targets_in_use = firstArray.targets_in_use
            view = com.getView()
            self.view = view

            # generate a unique variable name used on target representing this instance
            global arrayOfStructs_id
            self.name = 'arrayOfStructsObj' + str(arrayOfStructs_id)
            arrayOfStructs_id += 1

            # create an arrayOfStructsClass object consisting of the numpy arrays on the targets in use
            names_tree = makeTree_like(structOfArrays, lambda a: repr(a))

            view.push({'names_tree' : names_tree}, targets=self.targets_in_use)

            view.execute('''\
                structOfArrays = arrayOfStructs.makeTree_like(names_tree, lambda a_name: globals()[a_name])
                %s = arrayOfStructs.arrayOfStructs(structOfArrays)''' % self.name,\
                targets=self.targets_in_use)

        if onTarget == 'False' and isinstance(self, h5_ndarray):
            self.distaxis = firstArray.distaxis

    def __del__(self):
        if onTarget == 'False' and isinstance(self, ndarray):
            # delete remote arrayOfStructs object
            self.view.execute('del %s' % self.name, targets=self.targets_in_use)

    def __repr__(self):
        return self.name

    def __str__(self):
        print self.structOfArrays

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
    items = [item for item in treeItems(structOfArrays)]
    array_type = type(items[0][1])

    assert all(type(a) == array_type for name, a in items),\
        "all arrays in 'structOfArrays' must be of the same type"

    # create a new class type out of the arrayOfStructsClass that inherits from the arrays' type.
    my_arrayOfStructsClass = type("ArrayOfStructs_" + array_type.__module__ + "-" + array_type.__name__,\
        (array_type,), dict(arrayOfStructsClass.__dict__))

    if onTarget == 'True':
        # This is a workaround. Instanciation of my_arrayOfStructsClass raises an exception on target
        # before executing the constructor. No idea why.
        return arrayOfStructsClass(structOfArrays)
    else:
        return my_arrayOfStructsClass(structOfArrays)

# make 'arrayOfStructs' applicable for local arrays, like e.g. numpy arrays, on engines
if onTarget == 'False':
    view = com.getView()
    view.execute('from pyDive import arrayOfStructs')
