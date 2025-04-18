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

# Protect import of Plotting in case matplotlib not installed.
try:
    from . import Plotting
except ImportError:
    pass

# Protect import of ModelPlot in case matplotlib not installed.
try:
    from . import ModelPlot
except ImportError:
    pass

# We do it this way so it's easier to reload.
from . import Spectrum_mod

Spectrum = Spectrum_mod.Spectrum

from moments._version import __version__

# When doing arithmetic with Spectrum objects (which are masked arrays), we
# often have masked values which generate annoying arithmetic warnings. Here
# we tell numpy to ignore such warnings. This puts greater onus on the user to
# check results, but for our use case I think it's the better default.
import numpy

numpy.seterr(all="ignore")
