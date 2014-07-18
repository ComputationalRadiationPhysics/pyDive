import IPParallelClient as com
import sys
#import ndarray    # this import is done by the ndarray module itself due to circular dependencies
import ndarray_factories

def __prepare_operand(operand, target):
    if isinstance(operand, ndarray.ndarray):
        return operand.dist_like(target)
    else:
        return operand

def unary_op(a, op):
    result = ndarray_factories.hollow_like(a)
    view = com.getView()
    view.execute("%s = %s%s" % (repr(result), op, repr(a)), targets=result.targets_in_use)
    return result

def binary_op(lhs, rhs, op):
    rhs = __prepare_operand(rhs, lhs)

    result = ndarray_factories.hollow_like(lhs)
    view = com.getView()
    view.execute("%s = %s %s %s" % (repr(result), repr(lhs), op, repr(rhs)), targets=result.targets_in_use)
    return result

def binary_rop(rhs, lhs, op):
    lhs = __prepare_operand(lhs, rhs)

    result = ndarray_factories.hollow_like(rhs)
    view = com.getView()
    view.execute("%s = %s %s %s" % (repr(result), repr(lhs), op, repr(rhs)), targets=result.targets_in_use)
    return result

def binary_iop(lhs, rhs, iop):
    rhs = __prepare_operand(rhs, lhs)

    view = com.getView()
    view.execute("%s %s %s" % (repr(lhs), iop, repr(rhs)), targets=lhs.targets_in_use)
    return lhs

def n_ary_fun(fun, *args):
    a = args[0]

    args = [a.name] + [__prepare_operand(arg, a) for arg in args[1:]]
    args_str = ", ".join(args)

    result = ndarray_factories.hollow_like(a)
    view = com.getView()
    view.execute("%s = %s(%s)" % (repr(result), fun, args_str), targets=result.targets_in_use)
    #\todo: determine dtype from the result of fun and not from args[0]
    return result

class n_ary_fun_wrapper(object):
    def __init__(self, fun_name):
        self.fun_name = fun_name

    def __call__(self, *args):
        return n_ary_fun(self.fun_name, *args)

# math function names
trigonometric_funs = ('sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan')
hyperbolic_funs = ('sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh')
rounding_funs = ('around', 'round', 'rint', 'fix', 'floor', 'ceil', 'trunc')
exp_log_funs = ('exp', 'expm1', 'exp2', 'log', 'log10', 'log2', 'log1p')
misc_funs = ('abs', 'sqrt', 'maximum', 'minimum')
math_funs = trigonometric_funs + hyperbolic_funs + rounding_funs + exp_log_funs + misc_funs

# declare math functions in the namespace of this module
my_module = sys.modules[__name__]
for math_fun in math_funs:
    setattr(my_module, math_fun, n_ary_fun_wrapper(math_fun))
