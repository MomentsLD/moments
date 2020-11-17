===========================
The Site Frequency Spectrum
===========================

.. note::
   Some sections of this documentation are incomplete and only contain
   bullet-points to fill in. This will be completed in the very near
   future.

This page describes the Site Frequency Spectrum (SFS), how to compute
its expectation using ``moments``, manipulate spectra, implement demographic
models using the ``moments`` API, and computing and saving spectra from
a VCF.

The SFS
^^^^^^^

A site-frequency spectrum is a :math:`p`-dimensional histogram, where :math:`p`
is the number of populations for which we have data. Thus, the shape of the SFS
is :math:`(n_0+1) \times (n_1+1) \times \ldots (n_{p-1}+1)`, where :math:`n_i`
is the haploid sample size in population :math:`i`. An entry of the SFS
(call it ``fs``) stores the number, density, or probability for SNP frequencies
given by the index of that entry. That is, ``fs[j, k, l]`` is the number
(or density) of mutations with allele frequencies ``j`` in population 0, ``k``
in population 1, and ``l`` in population 2. (Note that all indexing, as is
typical in Python, is zero-based.)

Spectrum objects in ``moments``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SFS are stored as ``moments.Spectrum`` objects. If you are familiar with
`dadi <https://bitbucket.org/ryangutenkunst/dadi.git>`_'s Spectrum objects,
then you already will know your way around a ``moments.Spectrum`` object.
``moments`` has built off the ``dadi`` SFS construction, manipulation,
and demographic specification, with minor adjustments that reflect the
differences between the simulation engines and parameterizations.

``Spectrum`` objects are a subclass of ``numpy.masked_array``, so that standard
array manipulation is possible. Indexing also works the same way as a typical
array, so that ``fs[2, 3, 5]`` will return the entry in the SFS corresponding
to allele frequencies :math:`(2, 3, 5)` (here, in a three-population SFS).
Similarly, we can check if the SFS is masked at a given entry. For example,
``fs.mask[0, 0]`` returns whether the "fixed" bin (where no samples carry
the derived allele) is ignored.

A ``Spectrum`` object has a few additional useful attributes:

- ``fs.pop_ids``: A list of population IDs (as strings) for each population
  in the SFS.
- ``fs.sample_sizes``: A list of sample sizes (as integers) corresponding to
  the shape of the SFS.
- ``fs.folded``: If True, the SFS is folded, meaning we polarize
  allele frequencies by the minor allele frequency. If False, the SFS is
  polarized by the derived allele.

Manipulating SFS
^^^^^^^^^^^^^^^^

Along with standard array manipulations, there are operations specific to SFS.
Some of these are equivalent to standard array operations, but we ensure that
the masking and population IDs are updated properly.

Folding
=======

Folding a SFS removes information about how SNPs are polarized, so that the
Spectrum stores counts of mutations with a given minor allele frequency. To
fold a SFS, we call ``fold()``, which returns a folded Spectrum object.

.. jupyter-execute::
    :hide-code:

    import moments
    import numpy as np

For example, the standard neutral model of sample size 10,

.. jupyter-execute::
    
    fs = moments.Demographics1D.snm([10])
    fs

can be folded to the minor allele frequency, which updates the allele counts
in the minor allele frequency bins and the mask:

.. jupyter-execute::

    fs_folded = fs.fold()
    fs_folded

When folding multi-dimensional SFS, note that the folding occurs over the global
minor allele frequency.

Projecting
==========

SFS projection takes a Spectrum of some sample size and reduces the sample size
in one or more populations. The output Spectrum sums over all possible
down-samplings so that it is equivalent to having sampled a smaller sample size
to begin with.

.. jupyter-execute::
    
    fs_proj = fs.project([6])
    fs_proj

For multi-dimensional frequency spectra, we must pass a list of sample sizes
of equal length to the dimension of the SFS:

.. jupyter-execute::
    
    fs = moments.Spectrum(np.random.rand(121).reshape((11, 11)))
    fs_proj = fs.project([6, 4])
    fs_proj

Marginalizing
=============

Resampling
==========

Demographic events
^^^^^^^^^^^^^^^^^^

Population splits
=================

Admixture and mergers
=====================

Pulse migration
===============

Integration
^^^^^^^^^^^

- size functions
- integration time and time units
- migration rates
- scaled mutation rate
- selection and dominance
- frozen populations
- mutation models (ISM vs reversible mutations)

Demographic models
^^^^^^^^^^^^^^^^^^

- IM example
- see Gallery for more examples of 1-, 2-, and 3-population demographic models

Computing summary statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One-population stats

- Watterson's theta
- Diversity (pi)
- Tajima's D

Two-population stats

- FST

Computing SFS from a VCF
^^^^^^^^^^^^^^^^^^^^^^^^

Using ``Misc.make_data_dict_vcf`` and ``Spectrum.from_data_dict``.

Storing and loading data
^^^^^^^^^^^^^^^^^^^^^^^^

- The Spectrum file format
- Writing to file
- Loading from file
