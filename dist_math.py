import IPParallelClient as com
import sys

def unary_op(a, op):
    result = empty_like(a)
    view = com.getView()
    view.execute("%s = %s%s" % (result.name, op, a.name), targets=result.targets_in_use)
    return result

def binary_op(lhs, rhs, op):
    if isinstance(rhs, ndarray):
        rhs = rhs.dist_like(lhs)
        rhs_repr = rhs.name
    else:
        rhs_repr = repr(rhs)

    result = empty_like(lhs)
    view = com.getView()
    view.execute("%s = %s %s %s" % (result.name, lhs.name, op, rhs_repr), targets=result.targets_in_use)
    return result

def binary_rop(lhs, rhs, op):
    if isinstance(rhs, ndarray):
        rhs = rhs.dist_like(lhs)
        rhs_repr = rhs.name
    else:
        rhs_repr = repr(rhs)

    result = empty_like(lhs)
    view = com.getView()
    view.execute("%s = %s %s %s" % (result.name, rhs_repr, op, lhs.name), targets=result.targets_in_use)
    return result

def binary_iop(lhs, rhs, iop):
    if isinstance(rhs, ndarray):
        rhs = rhs.dist_like(lhs)
        rhs_repr = rhs.name
    else:
        rhs_repr = repr(rhs)

    view = com.getView()
    view.execute("%s %s %s" % (lhs.name, iop, rhs_repr), targets=lhs.targets_in_use)
    return lhs

def n_ary_fun(fun, *args):
    a = args[0]
    args_str = a.name

    args_repr = [repr(arg) for arg in args[1:]]
    for arg in args_repr:
        args_str += ',' + arg

    result = empty_like(a)
    view = com.getView()
    view.execute("%s = %s(%s)" % (result.name, fun, args_str), targets=result.targets_in_use)
    return result

class n_ary_fun_wrapper(object):
    def __init__(self, fun_name):
        self.fun_name = fun_name

    def __call__(self, *args):
        return n_ary_fun(self.fun_name, *args)

math_funs = ('abs', 'exp')

my_module = sys.modules[__name__]
for math_fun in math_funs:
    setattr(my_module, math_fun, n_ary_fun_wrapper(math_fun))
