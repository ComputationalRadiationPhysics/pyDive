.. pyDive documentation master file, created by
   sphinx-quickstart on Wed Jul 23 15:53:56 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyDive's documentation!
==================================

Contents:

.. toctree::
   :maxdepth: 4

   start
   tutorial
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. glossary::
   
   engine
    The cluster nodes of *IPython.parallel* are called *engines*. Sometimes they are also called *targets*.
    They are the workers of pyDive performing all the computation and file i/o and they hold the actual array-memory.
    From the user perspective you don't to deal with them directly.

