# conftest.py - the tests live in this folder but import the pipeline modules,
# which sit in the three sibling package directories. put those on sys.path once
# here (pytest loads conftest before collecting tests) so every test can import
# them. modules resolve their own cross-directory dependencies internally.

import os
import sys

_DA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Data Analysis
for _sub in ("Imputation and Smoothing",
             "Feature Engineering and Modeling",
             "Exploratory Data Analysis"):
    _p = os.path.join(_DA, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
