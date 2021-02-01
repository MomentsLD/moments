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

from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError as e:
    print("cython not installed, please install cython first")
    raise e

try:
    import numpy as np
except ImportError as e:
    print("numpy not installed, please install numpy first")
    raise e

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
    version=open("moments/_version.py").readlines()[-1].split()[-1].strip("\"'"),
    author="Aaron Ragsdale, Julien Jouganous, Simon Gravel, Ryan Gutenkunst",
    author_email="aaron.ragsdale@mail.mcgill.ca, simon.gravel@mcgill.ca",
    url="http://bitbucket.org/simongravel/moments",
    packages=["moments", "moments.Triallele", "moments.TwoLocus", "moments.LD"],
    license="MIT",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, language_level="3"),
    python_requires=">=3.6",
    install_requires=["numpy >=1.12.1", "cython >=0.25", "scipy >=1.3", "mpmath >=1.0"],
)
