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
__doc__ = None

from .. import IPParallelClient as com
import sys
#import ndarray    # this import is done by the ndarray module itself due to circular dependencies
import factories

def __prepare_operand(operand, target):
    if isinstance(operand, ndarray.ndarray):
        return operand.dist_like(target)
    else:
        return operand

def unary_op(a, op):
    result = factories.hollow_like(a)
    view = com.getView()
    view.execute("%s = %s%s" % (repr(result), op, repr(a)), targets=result.targets_in_use)
    return result

def binary_op(lhs, rhs, op):
    rhs = __prepare_operand(rhs, lhs)

    result = factories.hollow_like(lhs)
    view = com.getView()
    view.execute("%s = %s %s %s" % (repr(result), repr(lhs), op, repr(rhs)), targets=result.targets_in_use)
    return result

def binary_rop(rhs, lhs, op):
    lhs = __prepare_operand(lhs, rhs)

    result = factories.hollow_like(rhs)
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

    result = factories.hollow_like(a)
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

# module doc
__doc__ = \
"Mathematical functions supporting :class:`pyDive.ndarray.ndarray.ndarray`:\n\n" +\
"**trigonometric**:\n\n" + ", ".join(trigonometric_funs) + "\n\n" +\
"**hyperbolic**:\n\n" + ", ".join(hyperbolic_funs) + "\n\n" +\
"**rounding**:\n\n" + ", ".join(rounding_funs) + "\n\n" +\
"**exponential and logarithmic**:\n\n" + ", ".join(exp_log_funs) + "\n\n" +\
"**misc**:\n\n" + ", ".join(misc_funs)