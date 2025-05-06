============
Introduction
============

.. note::
    These docs are under development. In particular, many of the modules have not
    yet been completed and some of the extensions are not documented in great
    detail. If you find any issues, confusing bits, or have suggestions to make
    them more complete or clearer, please open an issue or a PR. Thanks!

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

- Jouganous, J., Long, W., Ragsdale, A. P., & Gravel, S. (2017). Inferring the
  joint demographic history of multiple populations: beyond the diffusion
  approximation. *Genetics*, 206(3), 1549-1567.

If you use ``moments.LD`` in your research, please cite:

- Ragsdale, A. P. & Gravel, S. (2019). Models of archaic admixture and recent
  history from two-locus statistics. *PLoS Genetics*, 15(6), e1008204.

- Ragsdale, A. P. & Gravel, S. (2020). Unbiased estimation of linkage
  disequilibrium from unphased data. *Mol Biol Evol*, 37(3), 923-932.


If you use ``moments.TwoLocus`` in your research, please cite:

- Ragsdale, A. P. (2022). Local fitness and epistatic effects lead to distinct
  patterns of linkage disequilibrium in protein-coding genes. *Genetics*,
  221(4), iyac097.


**********
Change log
**********

1.4.2
=====
- Introduce a new, flexible method to compute the SFS from a VCF file
- Add deprecation warnings to previous VCF parsing methods, which will be
  removed in version 1.5.0
- Remove deprecated ms and sfscode functions

1.4.1
=====
- Bug fix in 2D SFS plotting

1.4.0
=====
- Avoid masking zeros values in SFS when using Plotting functions
- Fix bug in mask_corners behavior in Spectrum.from_file()
- Fix bug in confidence interval calculation using Demes-based SFS inference
- Support python 3.10 through 3.13, require numpy >= 2.0

1.3.1
=====

1.3.0
=====

1.2.3
=====

- Invert meaning of rescaling parameter in ``moments.Demes.DemesUtil.rescale()``
  to match standard conventions.

1.2.0
=====

- Change project name to ``moments-popgen``, still imports as ``moments``
- Fix unexpected behavior when perturbing input parameters in Demes inference
- Change scaling behavior for mutation, selection and recombination  when
  computing statistics using Demes (detailed in documentation)
- Add ``H2()`` function to LD statistics

1.1.18
======

- Implement underdominance and overdominance in SFS integration.

1.1.17
======

- Allow selection and dominance to be specified as time-dependent functions
  in SFS integration.

1.1.16
======

- Flexible steady-state solution in LD statistics for multi-population
  island models
- Fix bug in reversible mutation model for SFS with multiple populations
  with migration

1.1.15
======

- Fix various bugs in LD parsing methods, including when data is missing and
  recursion errors in cythonized genotype calculation methods
- Add steady state solution to LD methods

1.1.14
======

- Fix bugs when computing multi-population LD statistics using phased haplotype
  data
- Steady state LD statistics for two-population island models

1.1.13
======

- Function to parse ANGSD-formatted data as a moments.Spectrum object (issue #106)
- Catch if genotype matrix is too large to compute pairwise LD (issue #105)

1.1.12
======

- Efficiency improvements in LD Parsing and Integration
- Test demes graph slicing features

1.1.11
======

- The LD inference methods now allow calculation of f-statistics (f2, f3, f4)
- Demes methods allow multiple sources in pulses
- Demes integration allow for ancient samples
- Fix bugs in L-BFGS-B methods for inference using the SFS

1.1.10
======

- Add warnings and exceptions if bins are improperly defined in LD.Parsing (Issue #99).
- Remove ``ld_extensions`` flag from installation so that all extensions are built
  automatically.
- Pin cython to ~0.29 until recursion error is fixed
- Allow samples to be specified with a dictionary for SFS calculation with Demes
- Memory-efficient caching of projection in TwoLocus
- Add LD inference using Demes and clean up uncertainty calculations for SFS inference
  using demes

1.1.9
=====

- Allow ancient samples in Demes inference function
- Add selection and dominance to Demes SFS integration function
- Add f2 and f4 statistics to LDstats object
- Allow multiple simultaneous merger events in Demes integration methods
- Add uncertainty functions to Demes SFS inference module
- Refactor Demes SFS inference options (#85)
- Add function to compute genotype matrix from the SFS
- Add function to compute allele frequency threshold LD statistics from
  TwoLocus spectrum
- Fix factor of 2 discrepancy between LD and TwoLocus mutation model (#60)

1.1.8
=====

- Fix bug that plotted multiple colorbars in plot_single_2d_sfs (issue #82).
- Add L-BFGS-B optimization method to LD inference.
- Fix bug in SFS inference using demes when a branch event time is a variable parameter.
- Fix bug in LD Godambe method that improperly normalized J matrix and cU vector.

1.1.7
=====

- Inference using demes allows for ancestral misidentification estimation
  (#81).
- Fst computation now has option for all pairwise computations (#80).
- Bug fix when computing LD with an input VCF that includes multiple
  chromosomes (#78).
- Bug fix when computing LD means over multiple regions.
- Expanded documentation, particularly for clarification of installation steps
  in docs when using LD parsing methods (#79), usage of Godambe methods for
  computing confidence intervals (#77), and more details for LD methods.

1.1.6
=====

- Many small bug fixes and API improvements to LD parsing, inference, and
  confidence interval methods.
- Expanded documentation for computing, parsing, and running inference using LD
  statistics (#73).
- Expand LD examples in repository and bring them up to date with current API
  (#74).
- Minor improvements to 1D SFS plotting (#64).

1.1.5
=====

- Use (chrom, pos) tuple as data dictionary key, to avoid conflicts with
  underscores. Underscores in contig/chromosome names are again supported.
- Add branch function to Spectrum class.
- Fix bug when computing SFS from demes with branches occurring simultaneously
  (#71).
- Fix bug when computing SFS from demes with pulses occurring simultaneously
  (#72).

1.1.4
=====

- Fix bugs in Plotting multi-population SFS comparisons that were showing each
  subplot in a new figure instead of in a single plot.
- Hide the intrusive scale bar in ModelPlot by default.

1.1.3
=====

- Fix bug in Misc.make_data_dict_vcf that skipped any site with missing data.
- Fix numpy deprecation warning when projecting.
- Documentation updates for miscellaneous functions.
- Fix bug where copying and pickling LDstats objects resulted in a recursion
  error (#66).

1.1.2
=====

- Fix bug when checking if matplotlib is installed for model plotting  (issue
  #68).
- Now compatible with demes >= 0.1.


1.1.1
=====

- Fix a pesky RecursionError in ``moments.LD.Inference.sigmaD2``.
- Fix bug when simulating LD using ``Demes`` if admixture timing coincides with
  a deme's end time.
- Fix ``numpy.float`` deprecation warning in ``moments.LD.Numerics``.
- Update demes methods to work with ``demes`` version 0.1.0a4.
- Improve (or at least change) some of the plotting outputs.
- Protect import of ``demes`` if not installed.


1.1.0
=====

- Completely rebuilt documentation, now hosted on [Read the
  Docs](https://moments.readthedocs.io/).
- Tutorials and modules in the documentation for running inference, inferring
  the DFE, and exploring LD under a range of selection models.
- More helpful documentation in docstrings.
- Support for
  [demes](https://moments.readthedocs.io/en/latest/extensions/demes.html).
- Simpler functions to improve Spectrum manipulation and demographic events,
  such as fs.split(), fs.admix, etc.
- API and numerics overhaul for Triallele and TwoLocus methods.
- Expanded selection models in the TwoLocus module.
- moments.LD methods are now zero-based.
- Reversible mutation model supports a single symmetric mutation rate.

1.0.9 
=====

- Numpy version bump from 0.19 to 0.20 creates incompatibility if cython
  extension are built with different version than user environment. This more
  explicitly specifies the numpy version to maintain compatibility (with thanks
  to Graham Gower).

1.0.8
=====

- Allow for variable migration rate by passing a function as the migration
  matrix (with thanks to Ekaterina Noskova/@noscode).
- Fixes an issue with ModelPlot when splitting 3D and 4D SFS.

1.0.7
=====

- Bug fixes and haplotype parsing in moments.LD.Parsing.
  (Issues #38 through #42, with thanks to Nathaniel Pope).


1.0.6
=====

- Updates to installation, so that ``pip`` installs dependencies automatically.
- Protect against importing ``matplotlib`` if not installed.
- ``Triallele`` and ``TwoLocus`` now ensure using CSC format sparse matrix to avoid
  sparse efficiency warnings.
- Streamline test suite, which now works with ``pytest``, as
  ``python -m pytests tests``.

1.0.5
=====

- Fixes install issues using pip: ``pip install .`` or
  ``pip install git+https://github.com/MomentsLD/moments.git`` is now functional.

1.0.4
=====

- Stable importing of scipy.optimize nnls function.
- Fixes a plotting bug when ax was set to None (from @noscode - thanks!).

1.0.3
=====

- Options in plotting scripts for showing and saving output.
- Add confidence interval computation for LD.
- Add parsing script for ANGSD frequency spectrum output.

Note that we started tracking changes between versions with version 1.0.2.

