# moments: tools for diversity statistics and inference

Please see the [documentation](https://moments.readthedocs.io/)
for more details, examples, and tutorials.

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

If you use `moments.TwoLocus` in your research, please cite:

- Ragsdale, A. P. (2021). Can we distinguish modes of selective interactions
  from linkage disequilibrium? BioRxiv, doi: https://doi.org/10.1101/2021.03.25.437004

`moments` is developed in [Simon Gravel's
group](http://simongravel.lab.mcgill.ca/Home.html) in the Human Genetics
department at McGill University, with with ongoing maintenance and development
by the Gravel Lab and [Aaron Ragsdale](http://apragsdale.github.io).

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
pip install numpy, cython, scipy, mpmath
pip install .
```

If you use `conda`, `moments` is available via `bioconda`:

```
conda config --add channels bioconda
conda install moments
```

### Dependencies and details

`moments` and `moments.LD` requires a number of dependencies. At a minimum,
these include

- numpy

- scipy

- cython

- mpmath

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
python setup.py build_ext -i
python setup.py install
```

You should then be able to import `moments` in your python scripts. Entering an
ipython or python session, type `import moments`. More details on installation
can be found in the
[documentation](https://moments.readthedocs.io/en/latest/installation.html).
If, for any reason, you have trouble installing moments after following these
steps, please submit an
[issue](https://bitbucket.org/simongravel/moments/issues).

If you use `Parsing` from `moments.LD`, which reads vcf files and computes LD
statistics to compare to predictions from `moments.LD`, you will need to
additionally install

- hdf5

- scikit-allel

## Changelog

All changes are detailed in the
[documentation](https://moments.readthedocs.io/en/latest/introduction.html#change-log).
