# moments: population genetic analyses and inference using diversity statistics

Please see the [documentation](https://momentsld.github.io/moments/)
for more details, examples, tutorials and API usage.

`moments` provides a suite of methods for demographic history and selection
inference from genetic data, based on diffusion approximations to the one- and
two-locus allele frequency spectrum. `moments` is modeled after the
[∂a∂i](https://bitbucket.org/gutenkunstlab/dadi/) open source package developed
by [Ryan Gutenkunst](http://gutengroup.mcb.arizona.edu). For SFS-based methdos,
we largely reuse `∂a∂i`'s API, but introduce a new simulation engine. This new
method is based on the direct computation of the frequency spectrum without
solving the diffusion system, removing the need for frequency grids as used in
`∂a∂i`. `moments.LD`, packaged within `moments`, implements methods for
computing linkage disequilibrium statistics and running multi-population
demographic inference using patterns of LD.

## Getting started

`moments` now supports Python 3, and we no longer guarantee compatibility with
Python 2.

The simplest way to install `moments` is using `pip`:

```
pip install moments-popgen
```

`moments` can then be imported using `import moments`. <b>Important note:</b>
`pip install moments` installs a different package named moments, and our
pypi package is named `moments-popgen`.

We can install the development branch directly from Github by running

```
pip install git+https://github.com/MomentsLD/moments.git@devel
```

Alternatively, you can clone the git repository to make an editable or
development build.

```
git clone https://github.com/MomentsLD/moments.git
```

and then from within the moments directory (`cd moments`), run

```
pip install -r requirements.txt
pip install .
```

If you use `conda`, `moments` is available via `bioconda`:

```
conda config --add channels bioconda
conda install moments
```

## Citing moments

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

`moments` is developed in the [Simon
Gravel](http://simongravel.lab.mcgill.ca/Home.html) and [Aaron
Ragsdale](https://apragsdale.github.io/) research groups, at McGill University and
UW-Madison, respectively. For any issues, questions or bug reports, please open
an [issue on Github](https://github.com/MomentsLD/moments/issues).

### Dependencies

If you install `moments` from source (e.g., after cloning the repository), you
will need to install the dependencies. These are all listed in
`requirements.txt`, and can be installed via `pip` after navigating to the
`moments` directory:

```
pip install -r requirements.txt
```

A few more details: `moments` and `moments.LD` requires a handful of
dependencies. At a minimum, these include

- `numpy`

- `scipy`

- `cython`

- `mpmath`

- `demes`

We also strongly recommend installing `ipython`.

If you are using `conda`, all dependencies can be installed by navigating to
the moments directory and then running

```
conda install --file requirements.txt
```

Once dependencies are installed, to install `moments`, run the following command
in the moments directory:

```
python -m pip install -e .
```

You should then be able to import `moments` in your python scripts. Entering an
`ipython` or python session, try to `import moments`. More details on
installation can be found in the
[documentation](https://momentsld.github.io/moments/installation.html). If, for
any reason, you have trouble installing moments after following these steps,
please submit an [issue](https://github.com/MomentsLD/moments/issues).

If you use `Parsing` from `moments.LD`, which reads VCF files and computes LD
statistics to compare to predictions from `moments.LD`, you will need to
additionally install

- `hdf5`

- `scikit-allel`

- `pandas`

## Changelog

All changes are detailed in the
[documentation](https://moments.readthedocs.io/en/latest/introduction.html#change-log).
