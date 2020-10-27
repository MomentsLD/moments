# Importing these adds a 'bdist_mpkg' option that allows building binary
# packages on OS X.
try:
    import setuptools
    import bdist_mpkg
except ImportError:
    pass

import sys

if "--ld_extensions" in sys.argv:
    build_ld_extensions = True
    sys.argv.remove("--ld_extensions")
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
    if arg.startswith("--compiler"):
        compiler = arg.split("=")[1]
if compiler in ["unix", "mingw32", "cygwin"]:
    extra_compile_args = []
    # RNG: This seems to cause problems on some machines. To test for
    # compatibility with VC++, uncomment this line.
    # extra_compile_args = ['-std="iso9899:1990"', '-pedantic-errors']
else:
    extra_compile_args = []

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

# cython extensions for moments
extensions = [
    Extension(
        "Jackknife",
        ["moments/Jackknife.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"],
    ),
    Extension(
        "LinearSystem_1D",
        ["moments/LinearSystem_1D.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"],
    ),
    Extension(
        "LinearSystem_2D",
        ["moments/LinearSystem_2D.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"],
    ),
    Extension(
        "Tridiag_solve",
        ["moments/Tridiag_solve.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"],
    ),
]

# cython extensions for moments.LD
if build_ld_extensions is True:
    extensions += [
        Extension(
            "genotype_calculations",
            ["moments/LD/genotype_calculations.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-w"],
        ),
        Extension(
            "genotype_calculations_multipop",
            ["moments/LD/genotype_calculations_multipop.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-w"],
        ),
        Extension(
            "sparse_tallying",
            ["moments/LD/sparse_tallying.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-w"],
        ),
    ]

setup(
    name="moments",
    # version='1.0.5',
    version=open("moments/_version.py").readlines()[-1].split()[-1].strip("\"'"),
    author="Aaron Ragsdale, Julien Jouganous, Simon Gravel, Ryan Gutenkunst",
    author_email="simon.gravel@mcgill.ca, aaron.ragsdale@mail.mcgill.ca",
    url="http://bitbucket.org/simongravel/moments",
    packages=["moments", "moments.Triallele", "moments.TwoLocus", "moments.LD"],
    package_data={"tests": ["IM.fs"]},
    license="MIT",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, language_level="3"),
    install_requires=["numpy >=1.10", "cython >=0.25", "scipy >=1.3", "mpmath >=1.0"],
)
