"""
Copyright 2015 Heiko Burau

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

from itertools import chain

class lean_expression():
    
    def __init__(self, obj, op=None, *args):
        self.obj = obj
        self.op = op
        self.args = args

    def isterminal(self):
        return self.op is None

    def __getattr__(self, name):
        print "__getattr__", name
        return getattr(self.obj, name)

    def __str__(self):
        def printExpr(expr, indent, result=""):
            if isinstance(expr, lean_expression):
                if expr.isterminal():
                    result += indent + "terminal: {}\n".format(expr.obj)
                else:
                    result += indent + "obj:\n"
                    result = printExpr(expr.obj, indent + "  ", result)
                    result += indent + "op: {}\n".format(expr.op)
                    result += indent + "args:\n"
                    result += "\n".join([printExpr(arg, indent + "  ") if isinstance(arg, lean_expression) else indent + "  " + str(arg) for arg in expr.args])
            else:
                result += indent + "terminal: {}\n".format(expr)
            
            return result

        return printExpr(self, "", "")

    def map(self, f):
        if self.isterminal():
            return lean_expression(f(self.obj))
        return lean_expression(self.obj.map(f), self.op, *[arg.map(f) if isinstance(arg, lean_expression) else f(arg) for arg in self.args])

    def __iter__(self):
        """Loop all terminal expressions"""
        if self.isterminal():
            return iter((self,))
        else:
            return chain(*[iter(arg) for arg in self.args if isinstance(arg, lean_expression)])

    def evaluate(self, f):
        """Call f for each expression recursively and return the result."""
        if self.isterminal():
            return self.obj
        return f(self.obj.evaluate(f), self.op, [arg.evaluate(f) if isinstance(arg, lean_expression) else arg for args in self.args])

class expression(lean_expression):
    def __init__(self, context, obj, op=None, *args):
        self.context = context
        lean_expression.__init__(self, obj, op, *args);

    def __getitem__(self, args):
        return expression(self.context, self.obj[args])

    def __setitem__(self, key, value):
        result = expression(self.context, self[key], "__setitem__", slice(None), value)
        self.context.add_expression(result)

    def __elementwise_op__(self, op, *args):
        return expression(self.context, self, op, *args)

# add special operations like __add__, __mul__, etc. to `expression`

binary_ops = ["add", "sub", "mul", "floordiv", "div", "mod", "pow", "lshift", "rshift", "and", "xor", "or"]

binary_iops = ["__i" + op + "__" for op in binary_ops]
binary_rops = ["__r" + op + "__" for op in binary_ops]
binary_ops = ["__" + op + "__" for op in binary_ops]
unary_ops = ["__neg__", "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__long__", "__float__", "__oct__", "__hex__"]
comp_ops = ["__lt__", "__le__", "__eq__", "__ne__", "__ge__", "__gt__"]

make_special_op = lambda op: lambda self, *args: self.__elementwise_op__(op, *args)

special_ops_dict = {op : make_special_op(op) for op in \
    binary_ops + binary_rops + binary_iops + unary_ops + comp_ops}

from types import MethodType

for name, func in special_ops_dict.items():
    setattr(expression, name, MethodType(func, None, expression))

