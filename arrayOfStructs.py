import onTarget
import sys
# check whether this code is executed on target or not
if not onTarget.onTarget:
    import IPParallelClient as com
    from ndarray.ndarray import ndarray as ndarray
import numpy as np

def replaceTree(tree, expression):
    # traverse tree
    for key, value in tree.items():
        if type(value) is dict:
            replaceTree(value, expression)
        else:
            tree[key] = expression(value)
    return tree

def visitTwoTrees(treeA, treeB, visitor):
    # traverse trees
    for key, valueA in treeA.items():
        valueB = treeB[key]
        if type(valueA) is dict:
            visitTwoTrees(valueA, valueB, visitor)
        else:
            visitor(treeA, treeB, key, valueA, valueB)
    return treeA, treeB

# tree generator object
def treeItems(tree):
    # traverse tree
    for key, value in tree.items():
        if type(value) is dict:
            treeItems(value)
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
        self.dtype = replaceTree(structOfArrays.copy(), lambda a: a.dtype)
        self.nbytes = sum(a.nbytes for name, a in items)
        self.structOfArrays = structOfArrays

        if not onTarget.onTarget and isinstance(self, ndarray):
            #assert all(a.targets_in_use == firstArray.targets_in_use for name, a in items),\
            #    "all ndarrays in structure-of-arrays ('structOfArrays') must have an identical 'targets_in_use' attribute"

            self.distaxis = firstArray.distaxis
            self.targets_in_use = firstArray.targets_in_use
            self.idx_ranges = firstArray.idx_ranges
            view = com.getView()
            self.view = com.getView()

            # generate a unique variable name used on target representing this instance
            global arrayOfStructs_id
            self.name = 'arrayOfStructsObj' + str(arrayOfStructs_id)
            arrayOfStructs_id += 1

            # create an arrayOfStructsClass object consisting of the numpy arrays on the targets in use
            names_tree = replaceTree(structOfArrays.copy(), lambda a: repr(a))
            view.push({'names_tree' : names_tree}, targets=self.targets_in_use)
            view.execute('''\
                structOfArrays = arrayOfStructs.replaceTree(names_tree, lambda a_name: globals()[a_name])
                %s = arrayOfStructs.arrayOfStructs(structOfArrays)''' % self.name,\
                targets=self.targets_in_use)

    def __del__(self):
        if not onTarget.onTarget and isinstance(self, ndarray):
            # delete remote arrayOfStructs object
            self.view.execute('del %s' % self.name, targets=self.targets_in_use)

    def __repr__(self):
        return self.name

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
            "number of arguments does not correspond to the dimension (%d)" % len(self.shape)

        result = replaceTree(self.structOfArrays.copy(), lambda a: a[args])

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
            "number of arguments does not correspond to the dimension (%d)" % len(self.shape)

        def doArrayAssignmentWithSlice(treeA, treeB, name, arrayA, arrayB):
            treeA[name][args] = arrayB

        visitTwoTrees(self.structOfArrays, other, doArrayAssignmentWithSlice)

def arrayOfStructs(structOfArrays):
    items = [item for item in treeItems(structOfArrays)]
    array_type = type(items[0][1])
    assert all(type(a) == array_type for name, a in items),\
        "all arrays in 'structOfArrays' must be of the same type"

    # create a new class type out of the arrayOfStructsClass that inherits from the arrays' type.
    my_arrayOfStructsClass = type(array_type.__name__ + "_ArrayOfStructs",\
        (array_type,), dict(arrayOfStructsClass.__dict__))

    return my_arrayOfStructsClass(structOfArrays)

# make 'arrayOfStructs' applicable for local arrays, like e.g. numpy arrays, on engines
if not onTarget.onTarget:
    view = com.getView()
    view.execute('import arrayOfStructs')
