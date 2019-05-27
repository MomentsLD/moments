
**moments: tools for demographic inference**

`moments` implements methods for demographic history and selection inference from genetic data, based on diffusion approximations to the allele frequency spectrum. `moments` is based on the  [∂a∂i](https://bitbucket.org/gutenkunstlab/dadi/) open source package developed by Ryan Gutenkunst [http://gutengroup.mcb.arizona.edu]. We largely reuse `∂a∂i`'s interface but introduced a new simulation engine. This new method is based on the direct computation of the frequency spectrum without solving the diffusion system. Consequently we circumvent the numerical PDE approximations and we get rid of the frequency grids used in `∂a∂i`.

If you use `moments` in your research, please cite: Jouganous, J., Long, W., Ragsdale, A. P., & Gravel, S. (2017). Inferring the joint demographic history of multiple populations: beyond the diffusion approximation. Genetics, 206(3), 1549-1567.

If you use `moments.LD` in your research, please cite: Ragsdale, A. P. & Gravel, S. (2018). Models of archaic admixture and recent history from two-locus statistics. BioRxiv, doi: 10.1101/489401. 

`moments` is developed in Simon Gravel's group in the Human Genetics department at McGill University [http://simongravel.lab.mcgill.ca/Home.html].

**Getting started**

`moments` now supports python 3. Because python is soon discontinuing support for python 2, we do not actively ensure that moments remains fully compatable with python 2, and strongly recommend using python 3.

`moments` requires a number of dependencies. These are

- numpy

- scipy

- cython

- mpmath

- matplotlib

Dependencies can be installed using pip. For example to install `cython`,

    pip install cython

Depending on the python distribution you use, it may be usefull to add the directory to `cython` in your python path.

We also strongly recommend installing ipython.

If you are using conda, all dependencies can be installed by navigating to the moments directory and then running

    conda install --file requirements.txt

Once dependencies are installed, to install `moments`, run the following command in the moments directory:

    sudo python setup.py install

You should then be able to import `moments` in your python scripts. Entering an ipython or python session, type `import moments`. If, for any reason, you have trouble installing moments after following these steps, please submit an [Issue](https://bitbucket.org/simongravel/moments/issues).

