============
Introduction
============

.. note::
    These docs are under development. If you find any issues, confusing bits, or
    have suggestions to make them more complete or clearer, please open an issue
    or a PR. Thanks!

Welcome to ``moments``! ``moments`` implements methods for inferring demographic
history and patterns of selection from genetic data, based on solutions to the
diffusion approximations to the site-frequency spectrum (SFS).
The SFS implementation and interface of ``moments`` is large based on the
`∂a∂i <https://bitbucket.org/gutenkunstlab/dadi/>`_ open
source package developed by `Ryan Gutenkunst <http://gutengroup.mcb.arizona.edu>`_.
We largely reuse ``∂a∂i``'s interface but introduced a new simulation engine. This
new method is based on the direct computation of the frequency spectrum without
solving the diffusion system. Consequently we circumvent the numerical PDE
approximations and we get rid of the frequency grids used in ``∂a∂i``.

``moments.LD`` implements methods for computing linkage disequilibrium statistics
and running multi-population demographic inference using patterns of LD. This
extension contains methods for parsing phased or unphased sequencing data to
compute LD-decay for a large number of informative two-locus statistics, and
then uses those statistics to infer demographic history for large numbers of
populations.

``moments`` was developed in
`Simon Gravel's group <http://simongravel.lab.mcgill.ca/Home.html>`_ in the Human
Genetics department at McGill University, with maintenance and development by the
Gravel Lab and `Aaron Ragsdale <http://apragsdale.github.io>`_.

*********
Citations
*********

If you use ``moments`` in your research, please cite:

- Jouganous, J., Long, W., Ragsdale, A. P., & Gravel, S. (2017). Inferring the joint
  demographic history of multiple populations: beyond the diffusion approximation.
  Genetics, 206(3), 1549-1567.

If you use ``moments.LD`` in your research, please cite:

- Ragsdale, A. P. & Gravel, S. (2019). Models of archaic admixture and recent history
  from two-locus statistics. PLoS Genetics, 15(6), e1008204.

- Ragsdale, A. P. & Gravel, S. (2020). Unbiased estimation of linkage disequilibrium
  from unphased data. Mol Biol Evol, 37(3), 923-932.

**********
Change log
**********

1.0.6
=====

- Updates to installation, so that ``pip`` installs dependencies automatically

- Protect against importing ``matplotlib`` if not installed

- ``Triallele`` and ``TwoLocus`` now ensure using CSC format sparse matrix to avoid
  sparse efficiency warnings

- Streamline test suite, which now works with ``pytest``, as
  ``python -m pytests tests``

1.0.5
=====

- Fixes install issues using pip: ``pip install .`` or
  ``pip install git+https://bitbucket.org/simongravel/moments.git`` is now functional

1.0.4
=====

- Stable importing of scipy.optimize nnls function

- Fixes a plotting bug when ax was set to None (from @noscode - thanks!)

1.0.3
=====

- Options in plotting scripts for showing and saving output

- Add confidence interval computation for LD

- Add parsing script for ANGSD frequency spectrum output

Note that we started tracking changes between versions with version 1.0.2.

