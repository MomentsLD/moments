from . import (
    Numerics,
    Util,
    Matrices,
    Demographics1D,
    Demographics2D,
    Demographics3D,
    Inference,
    Godambe,
)

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