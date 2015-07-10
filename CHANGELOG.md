Changelog for pyDive
====================

1.2.2
-----
**Date:** 2015-07-10

 - add `setup_requires` to setup.py

1.2.1
-----
**Date:** 2015-07-10

 - add `arrays.local` package to setup.py

1.2
---
**Date:** 2015-06-22

**New Features:**
 - arrays can be distributed along multiple axes
 - new distributed array: `pyDive.arrays.gpu_ndarray`. Distributed version of a pycuda array with extra support
   for non-contiguous memory (needed for working with sliced arrays).
 - `gather()` is called implicitly when the user wants to access an attribute, e.g. datapointer, which only
   the local array has.
 - implement indexing by bitmask

**Bug Fixes:**
 - integer indices were not wrapped in `__getitem__()`
 - fix exception in picongpu.py if dataset has no attributes

**Misc**-
 - `distaxes` parameter defaults to 'all'
 - all local array classes are located in pyDive/arrays/local
 - rename "arrayOfStructs" module to "structured"

1.1
---
**Date:** 2015-03-29

**New Features:**
 - adios support
 - implement `pyDive.fragment`
 - distributed array classes are auto-generated from local array class.

**Misc:**
 - restructure project files
 - optimize `pyDive.ndarray`

1.0.2
-----
**Date:** 2014-11-06

 - add MANIFEST.in

1.0.1
-----
**Date:** 2014-11-06

 - fix setup.py

1.0
---
**Date:** 2014-11-05

Development status: **beta**
