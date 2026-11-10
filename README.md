# moments: population genetic analyses and inference using diversity statistics

Please see the [documentation](https://momentsld.github.io/moments/)
for more details, examples, tutorials and API usage.

`moments` provides a suite of methods for demographic history and selection
inference from genetic data, based on diffusion approximations to the one- and
two-locus allele frequency spectrum. `moments` is modeled after the
[∂a∂i](https://github.com/RyanGutenkunst/dadi) open source package developed
by [Ryan Gutenkunst](https://gutengroup.arizona.edu/). For SFS-based methods,
we largely reuse `∂a∂i`'s API, but introduce a new simulation engine. This new
method is based on the direct computation of the frequency spectrum without
solving the diffusion system, removing the need for frequency grids as used in
`∂a∂i`. `moments.LD`, packaged within `moments`, implements methods for
computing linkage disequilibrium statistics and running multi-population
demographic inference using patterns of LD.

## Getting started

`moments` requires Python 3.11 or newer (up to 3.15).

The simplest way to install `moments` is using `pip`:

```
pip install moments-popgen
```

`moments` can then be imported using `import moments`.

<b>Important note:</b>
`pip install moments` installs a different package named moments, and our
pypi package is named `moments-popgen`.

`moments` is also available from [Bioconda](https://anaconda.org/bioconda/moments):

```
conda install -c conda-forge -c bioconda moments
```

We can install the development branch directly from Github by running

```
pip install git+https://github.com/MomentsLD/moments.git@devel
```

Alternatively, you can clone the git repository and install from source (use
`pip install -e .` for an editable, development install):

```
git clone https://github.com/MomentsLD/moments.git
cd moments
pip install .
```

### Dependencies

Installing with `pip` installs the required dependencies automatically:

- `numpy` (>= 2.0, < 3.0)
- `scipy` (>= 1.13)
- `cython` (>= 3.1)
- `mpmath` (>= 1.0, < 1.4)
- `demes` (>= 0.2)

Some features need additional packages:

- Plotting: `matplotlib`
- Parsing VCFs with `moments.LD.Parsing`: `scikit-allel`, `h5py`, and `pandas`

We also recommend installing `ipython` for interactive analyses.

For development, `requirements.txt` lists the dependencies:

```
pip install -r requirements.txt
```

or, with `conda`:

```
conda install -c conda-forge --file requirements.txt
```

To build the documentation locally, install `moments` along with the packages in
`requirements/docs.txt`, and then run `make html` in the `docs` directory:

```
python -m pip install . -r requirements/docs.txt
cd docs
make html
```

More details on installation can be found in the
[documentation](https://momentsld.github.io/moments/installation.html). If you
have trouble installing `moments`, please open an
[issue](https://github.com/MomentsLD/moments/issues).

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

- Ragsdale, A. P. (2022). Local fitness and epistatic effects lead to distinct
  patterns of linkage disequilibrium in protein-coding genes. Genetics, 221(4),
  iyac097.

If you use the Godambe-based uncertainty methods (in `moments.Godambe` or `moments.LD.Godambe`), 
please cite:

- Coffman, A. J., Hsieh, P. H., Gravel, S., & Gutenkunst, R. N. (2016).
  Computationally efficient composite likelihood statistics for demographic
  inference. Molecular biology and evolution, 33(2), 591-593.

`moments` is developed in the [Simon
Gravel](https://gravellab.github.io/) and [Aaron
Ragsdale](https://apragsdale.github.io/) research groups, at McGill University and
UW-Madison, respectively. For any issues, questions or bug reports, please open
an [issue on Github](https://github.com/MomentsLD/moments/issues).

## Changelog

All changes are detailed in the
[documentation](https://momentsld.github.io/moments/introduction.html#change-log).
