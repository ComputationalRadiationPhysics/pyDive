.. _pyDive.ndarray:

pyDive.arrays.ndarray module
============================

.. note:: All of this module's functions and classes are also directly accessable from the :mod:`pyDive` module.

pyDive.ndarray class
--------------------

.. autoclass:: pyDive.ndarray
    :members: __init__, gather, copy, dist_like

Factory functions
-----------------
These are convenient functions to create a *pyDive.ndarray* instance.

.. automodule:: pyDive.arrays.ndarray
    :members: array, empty, empty_like, hollow, hollow_like, zeros, zeros_like, ones, ones_like

Universal functions
-------------------

*numpy* knows the so called *ufuncs* (universal function). These are functions which can be applied
elementwise on an array, like *sin*, *cos*, *exp*, *sqrt*, etc. All of these *ufuncs* from *numpy* are
also available for *pyDive.ndarray* arrays, e.g. ::
    
    a = pyDive.ones([100])
    a = pyDive.sin(a)

