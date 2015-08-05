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

from expression import expression
from IPython.parallel import interactive, require
import pyDive.IPParallelClient as com

class lazy_context(object):

    def __init__(self, evaluator, arrays_dicts):
        self.evaluator = evaluator
        self.arrays_dicts = arrays_dicts
        self.expressions = []

    def add_expression(self, expr):
        self.expressions.append(expr)

    def __enter__(self):
        for arrays_dict in self.arrays_dicts:
            for key, value in arrays_dict.items():
                if hasattr(value, "dist_like"):
                    arrays_dict[key] = expression(self, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.evaluate()

        for arrays_dict in self.arrays_dicts:
            for key, value in arrays_dict.items():
                if isinstance(value, expression) and value.isterminal():
                    arrays_dict[key] = value.obj

    def evaluate(self):
        """Evaluate all expressions"""

        @require(self.evaluator.__module__)
        def local_evaluate(expr, evaluator_name):
            evaluator = eval(evaluator_name)
            evaluator(expr)

        print self.expressions

        for expr in self.expressions:
            root_array = expr.obj
            assert hasattr(root_array, "dist_like"), "The root array of an expression has to be distributed."

            # first, equalize distribution of all arrays of the expression
            for terminal_expr in expr:
                terminal_expr.obj = terminal_expr.obj.dist_like(root_array)

            # replace all terminals by their representation. For distributed arrays it's the local name.
            local_expr = expr.map(repr)

            # evaluate expression on engine
            view = com.getView()
            tmp_targets = view.targets # save current target list
            view.targets = root_array.decomposition.ranks
            view.apply(interactive(local_evaluate), local_expr, self.evaluator.__module__ + "." + self.evaluator.__name__)
            view.targets = tmp_targets # restore target list


def lazy_evaluation(evaluator, *arrays_dicts):
    return lazy_context(evaluator, arrays_dicts)

