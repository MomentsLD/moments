"""
For examples of moments' usage, see the examples directory in the source
distribution.

Documentation of all methods can be found in doc/api/index.html of the source
distribution.
"""

import logging

logging.basicConfig()

from . import Demographics1D
from . import Demographics2D
from . import Demographics3D
from . import Godambe
from . import Inference
from . import Integration
from . import Integration_nomig
import Jackknife
import LinearSystem_1D
import LinearSystem_2D
from . import Manips
from . import Misc
from . import Numerics
from . import Demes
from . import Parsing

# Protect import of ModelPlot in case matplotlib not installed.
try:
    from . import ModelPlot
except ImportError:
    pass

# We do it this way so it's easier to reload.
from . import Spectrum_mod

Spectrum = Spectrum_mod.Spectrum

from moments._version import __version__


# Plotting needs matplotlib, which is slow to import (and optional), so it is imported
# the first time it's used, as moments.Plotting, instead of with moments.
def __getattr__(name):
    if name == "Plotting":
        # (import_module, not "from . import Plotting", which would call this
        # function again to check for the attribute)
        import importlib

        return importlib.import_module(f"{__name__}.Plotting")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ["Plotting"])
