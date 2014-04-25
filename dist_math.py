import IPParallelClient as com

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

def binary_iop(lhs, rhs, iop):
    if isinstance(rhs, ndarray):
        rhs = rhs.dist_like(lhs)
        rhs_repr = rhs.name
    else:
        rhs_repr = repr(rhs)

    view = com.getView()
    view.execute("%s %s %s" % (lhs.name, iop, rhs_repr), targets=lhs.targets_in_use)
    return lhs
