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

# Protect import of Parsing in case dependencies are not installed.
# try:
#    from . import Parsing
# except ImportError:
#    pass

from . import Parsing

from . import LDstats_mod

LDstats = LDstats_mod.LDstats


# Plotting needs matplotlib, which is slow to import (and optional), so it is imported
# the first time it's used, as moments.LD.Plotting, instead of with moments.LD.
def __getattr__(name):
    if name == "Plotting":
        # (import_module, not "from . import Plotting", which would call this
        # function again to check for the attribute)
        import importlib

        return importlib.import_module(f"{__name__}.Plotting")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ["Plotting"])
