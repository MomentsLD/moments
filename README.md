
**moments: tools for demographic inference**

`moments` implements methods for demographic history and selection inference from genetic data, based on diffusion approximations to the allele frequency spectrum. `moments` is based on the  [∂a∂i](https://bitbucket.org/gutenkunstlab/dadi/) open source package developed by Ryan Gutenkunst [http://gutengroup.mcb.arizona.edu]. We reused ∂a∂i's interface but introduced a new spectrum simulation engine. This new method is based on the direct computation of the frequency spectrum without solving the PDE diffusion system. Consequently we make less approximations and we get rid of the frequency grids used in ∂a∂i. This approach is particularly efficient for multiple populations models (up to 5 populations).   

`moments` is developed in Simon Gravel's group in the population genetics department at McGill University [http://simongravel.lab.mcgill.ca/Home.html].

**Getting started**

`moments` uses cython for performances improvement, it can be installed using pip:

	pip install cython
Depending on the python distribution you use, it may be usefull to add the directory to `cython` in your python path.

To install `moments`, run the following command in the repository containing the sources:

	sudo python setup.py install

You must then be able to import `moments` in your python scripts.

