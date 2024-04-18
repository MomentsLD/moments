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
    url="https://github.com/MomentsLD/moments",
    packages=[
        "moments",
        "moments.Triallele",
        "moments.TwoLocus",
        "moments.LD",
        "moments.Demes",
    ],
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions, language_level="3"),
)
