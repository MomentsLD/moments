try:
    import allel
    imported = 1
except ImportError:
    imported = 0

import moments.LD

def check_allel_import():
    if imported == 0:
        raise("Did not load allel package needed for Parsing. Is allel installed?")


