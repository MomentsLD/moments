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

Projecting
==========

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
