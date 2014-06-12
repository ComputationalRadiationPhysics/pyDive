import IPParallelClient as com
import numpy as np

cloned_ndarray_id = 0

class cloned_ndarray(object):
    def __init__(self, shape, dtype=np.float, targets_in_use='all', no_allocation=False):
        self.shape = list(shape)
        self.dtype = dtype
        self.nbytes = self.dtype.itemsize * np.prod(self.shape)
        self.targets_in_use = targets_in_use
        self.view = com.getView()

        if self.targets_in_use == 'all':
            self.targets_in_use = list(self.view.targets)

        # generate a unique variable name used on the target representing this instance
        global cloned_ndarray_id
        self.name = 'cloned_ndarray' + str(cloned_ndarray_id)
        cloned_ndarray_id += 1

        if no_allocation:
            self.view.push({self.name : None}, targets=self.targets_in_use)
        else:
            self.view.push({'myshape' : self.shape, 'dtype' : self.dtype}, targets=self.targets_in_use)
            self.view.execute('%s = empty(myshape, dtype=dtype)' % self.name, targets=self.targets_in_use)

    def __del__(self):
        self.view.execute('del %s' % self.name, targets=self.targets_in_use)

    def __repr__(self):
        return self.name

    def reduce(self, op):
        result = self.view.pull(self.name, targets=self.targets_in_use[0])
        for target in self.targets_in_use[1:]:
            result = op(result, self.view.pull(self.name, targets=target))
        return result

    def sum(self):
        return self.reduce(lambda x, y: x+y)
