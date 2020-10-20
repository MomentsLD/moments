from . import (
    Numerics,
    Util,
    Matrices,
    Demographics1D,
    Demographics2D,
    Inference,
    Godambe,
)  # , Demographics3D, Corrections

# Protect import of Demography in case networkx is not installed.
try:
    from . import Demography
except ImportError:
    pass

# Protect import of Plotting in case matplotlib is not installed.
try:
    from . import Plotting
except ImportError:
    pass

# Protect import of Parsing in case dependencies are not installed.
# try:
#    from . import Parsing
# except ImportError:
#    pass

from . import Parsing

from . import LDstats_mod

LDstats = LDstats_mod.LDstats
