# Importing these adds a 'bdist_mpkg' option that allows building binary
# packages on OS X.
try:
    import setuptools
    import bdist_mpkg
except ImportError:
    pass

import os,sys

import numpy.distutils.core as core

if '--ld_extensions' in sys.argv:
    build_ld_extensions = True
    sys.argv.remove('--ld_extensions')
else:
    build_ld_extensions = False

#
# Microsoft Visual C++ only supports C up to the version iso9899:1990 (C89).
# gcc by default supports much more. To ensure MSVC++ compatibility when using
# gcc, we need to add extra compiler args. This code tries to ensure such
# arguments are added *only* when we're using gcc.
#
import numpy.distutils
compiler = numpy.distutils.ccompiler.get_default_compiler()
for arg in sys.argv:
    if arg.startswith('--compiler'):
        compiler = arg.split('=')[1]
if compiler in ['unix','mingw32','cygwin']:
    extra_compile_args = []
    # RNG: This seems to cause problems on some machines. To test for
    # compatibility with VC++, uncomment this line.
    #extra_compile_args = ['-std="iso9899:1990"', '-pedantic-errors']
else:
    extra_compile_args = []

from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

# cython extensions for moments
extensions = [
              Extension("Jackknife", ["moments/Jackknife.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"]),
              Extension("LinearSystem_1D", ["moments/LinearSystem_1D.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"]),
              Extension("LinearSystem_2D", ["moments/LinearSystem_2D.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"]),
              Extension("Tridiag_solve", ["moments/Tridiag_solve.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"])
              ]

setup(
      cmdclass = {'build_ext':build_ext},
      ext_modules = cythonize(extensions),
      )

# cython extensions for moments.LD
if build_ld_extensions is True:
    extensions = [
                  Extension("genotype_calculations", ["moments/LD/genotype_calculations.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"]),
                  Extension("genotype_calculations_multipop", ["moments/LD/genotype_calculations_multipop.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"]),
                  Extension("sparse_tallying", ["moments/LD/sparse_tallying.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-w"])
                  ]

    setup(
          cmdclass = {'build_ext':build_ext},
          ext_modules = cythonize(extensions),
          )

numpy.distutils.core.setup(name='moments',
                           #version='1.0.2',
                           # this is ugly, and might want to change, but version number is now stored in
                           # moments/_version.py and read from there
                           version=open("moments/_version.py").readlines()[-1].split()[-1].strip("\"'"),
                           author='Simon Gravel, Ryan Gutenkunst, Julien Jouganous, Aaron Ragsdale',
                           author_email='simon.gravel@mcgill.ca',
                           url='http://simongravel.lab.mcgill.ca/Home.html',
                           packages=['moments', 'moments.Triallele', 'moments.TwoLocus', 'moments.LD'],
                           package_data = {'tests':['IM.fs']},
                           license='BSD'
                           )
