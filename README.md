# moments: tools for demographic inference

`moments` implements methods for demographic history and selection inference
from genetic data, based on diffusion approximations to the allele frequency spectrum.
`moments` is based on the  [∂a∂i](https://bitbucket.org/gutenkunstlab/dadi/) open
source package developed by [Ryan Gutenkunst](http://gutengroup.mcb.arizona.edu).
We largely reuse `∂a∂i`'s interface but introduced a new simulation engine. This
new method is based on the direct computation of the frequency spectrum without
solving the diffusion system. Consequently we circumvent the numerical PDE
approximations and we get rid of the frequency grids used in `∂a∂i`.

`moments.LD` implements methods for computing linkage disequilibrium statistics and
running multi-population demographic inference using patterns of LD.

If you use `moments` in your research, please cite:

- Jouganous, J., Long, W., Ragsdale, A. P., & Gravel, S. (2017). Inferring the joint
  demographic history of multiple populations: beyond the diffusion approximation.
  Genetics, 206(3), 1549-1567.

If you use `moments.LD` in your research, please cite:

- Ragsdale, A. P. & Gravel, S. (2019). Models of archaic admixture and recent history
  from two-locus statistics. PLoS Genetics, 15(6), e1008204.

- Ragsdale, A. P. & Gravel, S. (2020). Unbiased estimation of linkage disequilibrium
  from unphased data. Mol Biol Evol, 37(3), 923-932.

`moments` is developed in
[Simon Gravel's group](http://simongravel.lab.mcgill.ca/Home.html) in the Human
Genetics department at McGill University, with maintenance and development by the
Gravel Lab and [Aaron Ragsdale](http://apragsdale.github.io).

## Getting started

`moments` now supports python 3. Because python is soon discontinuing support for
python 2, we do not actively ensure that moments remains fully compatable with python
2, and strongly recommend using python 3.

The simplest way to install `moments` is using `pip`. Note that `numpy` and `cython`
are install requirements, but installing `moments` directly from the git repository
using `pip` should install these dependencies automatically:

```
pip install git+https://bitbucket.org/simongravel/moments.git
```

Alternatively, you can clone the git repository

```
git clone https://bitbucket.org/simongravel/moments.git
```

and then from within the moments directory (`cd moments`), run

```
pip install numpy, cython
pip install .
```

Coming soon: `moments` will be available via `bioconda` in the near future.

### Dependencies and details

`moments` and `moments.LD` requires a number of dependencies. These are

- numpy

- scipy

- cython

- mpmath

- matplotlib

- networkx

- pandas

Dependencies can be installed using pip. For example to install `cython`,

```
pip install cython
```

Depending on the python distribution you use, it may be useful to add the directory
to `cython` in your python path.

We also strongly recommend installing `ipython`.

If you are using conda, all dependencies can be installed by navigating to the
moments directory and then running

```
conda install --file requirements.txt
```

or, alternatively,

```
pip install -r requirements.txt
```

Once dependencies are installed, to install `moments`, run the following command
in the moments directory:

```
sudo python setup.py install
```

You should then be able to import `moments` in your python scripts. Entering an
ipython or python session, type `import moments`. If, for any reason, you have
trouble installing moments after following these steps, please submit an
[Issue](https://bitbucket.org/simongravel/moments/issues).

If you use `Parsing` from `moments.LD`, which reads vcf files and computes LD
statistics to compare to predictions from `moments.LD`, you will need to
additionally install

- hdf5

- scikit-allel

## Changelog

### 1.0.9

- Numpy version bump from 0.19 to 0.20 creates incompatibility if cython extension
  are built with different version than user environment. This more explicitly
  specifies the numpy version to maintain compatibility (with thanks to Graham Gower)

### 1.0.8

- Allow for variable migration rate by passing a function as the migration matrix
  (with thanks to Ekaterina Noskova/noscode)

- Fixes an issue with ModelPlot when splitting 3D and 4D SFS

### 1.0.7

- Bug fixes and haplotype parsing in `moments.LD.Parsing`. (Issues #38-42,
  with thanks to Nathaniel Pope)

### 1.0.6

- Updates to installation, so that `pip` installs dependencies automatically

- Protect against importing `matplotlib` if not installed

- `Triallele` and `TwoLocus` now ensure using CSC format sparse matrix to avoid
  sparse efficiency warnings

- Streamline test suite, which now works with `pytest`, as
  `python -m pytests tests`

### 1.0.5

- Fixes install issues using pip: `pip install .` or
  `pip install git+https://bitbucket.org/simongravel/moments.git` is now functional

### 1.0.4

- Stable importing of scipy.optimize nnls function

- Fixes a plotting bug when ax was set to None (from noscode)

### 1.0.3

- Options in plotting scripts for showing and saving output

- Add confidence interval computation for LD

- Add parsing script for ANGSD frequency spectrum output

Started tracking changes between versions with version 1.0.2.
