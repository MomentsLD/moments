import numpy as np  
import os, gzip, scipy
import pandas
import pickle
import h5py 

from distutils.version import LooseVersion
try:
    import allel
    imported = 1
    assert LooseVersion(allel.__version__) >=  LooseVersion('1.2.0'), \
                "Parsing may not work with this version of scikit-allel, requires 1.2.0"
except ImportError:
    imported = 0

import Util

def check_allel_import():
    if imported == 0:
        raise("Did not load scikit-allel package needed for Parsing. Is allel installed?")


def read_vcf_to_h5(vcf_file_path):
    """
    reads vcf and writes to hdf5 format using allel package
    """
    check_allel_import()
    h5_file_path = vcf_file_path.split('vcf')[0] + '.h5'
    try:
        callset = h5py.File(h5_file_path, mode='r')
    except OSError:
        allel.vcf_to_hdf5(vcf_file_path, h5_file_path, fields='*', overwrite=True)
        callset = h5py.File(h5_file_path, mode='r')
    return callset

def get_stats(vcf_file_path=None, bin_edges=None, bed_file_path=None):
    pass
    

def bootstrap_regions()
    pass

