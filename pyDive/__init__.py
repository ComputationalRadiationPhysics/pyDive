# -*- coding: utf-8 -*-
__version__ = "0.1"

import os
onTarget = os.environ.get("onTarget", 'False')

# import only if this code is not executed on engine
if onTarget == 'False':
    from pyDive import *