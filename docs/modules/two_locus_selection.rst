.. _two-locus-usage:

=====================
Selection at two loci
=====================

.. todo:: This module has not been completed

Most users of ``moments`` will be using this package for computing the SFS and comparing
it to data. However, ``moments`` can do much more, such as computing expectations for
LD under complex demography, or triallelic or two-locus frequency spectra. Here, we'll
explore what we can do with the two-locus methods available in ``moments.TwoLocus``.

.. jupyter-execute::

    import moments.TwoLocus as motl

The two-locus allele frequency spectrum
=======================================

Similar to the single-site SFS, the two-locus frequency spectrum stores the number (or
density) of pairs of loci with given two-locus haplotype counts. Suppose the left locus
permits alleles `A`/`a` and the right locus permits `B`/`b`, so that there are four possible
haplotypes: (`AB`, `Ab`, `aB`, and `ab`). In a sample size of `n` haploid samples, we
observe some number of each haplotype, :math:`n_{AB} + n_{Ab} + n_{aB} + n_{ab} = n`. The
two-locus frequency spectrum stores the observed number of pairs of loci with each possible
sampling configuration, so that :math:`\Psi_n(i, j, k)` is the number (or density) of pairs
of loci with `i` type `AB`, `j` type `Ab`, and `k` type `aB`.

``moments.TwoLocus`` lets us compute the expectation of :math:`\Psi_n` for
single-population demographic scenarios, allowing for population size changes over time,
as well as arbitrary recombination distance separating the two loci and selection at
one or both loci. 
