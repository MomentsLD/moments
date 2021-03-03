 .. _sec_ld:

======================
Linkage Disequilibrium
======================

Using moment equations for the two-locus haplotype distribution, ``moments.LD`` lets
us compute a large family of linkage disequilibrium statistics in models with
arbitrary mutation and recombination rates and flexible demographic history with any
number of populations. The statistics are stored in a different way that the SFS, but
much of the API for implementing demographic events and integration is largely
consistent between the SFS and LD methods.

If you use ``moments.LD`` in your research, please cite:

- Ragsdale, A. P. & Gravel, S. (2019). Models of archaic admixture and recent history
  from two-locus statistics. *PLoS Genetics*, 15(6), e1008204.

- Ragsdale, A. P. & Gravel, S. (2020). Unbiased estimation of linkage disequilibrium
  from unphased data. *Mol Biol Evol*, 37(3), 923-932.


*************
LD statistics
*************

The LD statistics that ``moments.LD`` computes are low-order summaries of expected
LD between pairs of loci. In particular, we compute :math:`\mathbb{E}[D^2]`, the
expectation of the numerator of the familiar `r^2` measure of LD. From this system of
equations, we also compute :math:`\mathbb{E}[Dz] = \mathbb{E}[D(1-2p)(1-2q)]`, where
:math:`p` and :math:`q` are the allele frequencies at the left and right loci,
respectively; and we also compute :math:`\pi_2=\mathbb{E}[p(1-p)q(1-q)]`, a measure
of the "joint heterozygosity" of the two loci [Hill1968]_.

These statistics are stored in a list of arrays, where each list element corresponds
to a given recombination rate, :math:`\rho = 4N_er`, where `r` is the recombination
probability separating loci. The length of the list is the length of the number of
recombination rates given, plus one, as the last entry stores the single-locus
expected heterozygosity:

.. jupyter-execute::

    import moments.LD
    theta = 0.001 # the mutation rate 4*Ne*u
    rho = [0, 1, 10] # recombination rates 4*Ne*r between loci
    y = moments.LD.Demographics1D.snm(rho=rho, theta=theta) # steady-state expectations
    y

Here, we can see the decay of LD with increasing recombination rate, and also that
the heterozygisty equals the scaled mutation rate at steady-state, as expected.
On any LD object, we can get the list of statistics present by calling:

.. jupyter-execute::
    
    y.names()

The underscores index the populations for that statistic, so ``DD_0_0`` represents
:math:`\mathbb{E}[D_0 D_0] = \mathbb{E}[D_0^2]`, ``Dz_0_0`` represents
:math:`\mathbb{E}[D_0(1-2p_0)(1-2q_0)]`, and so on.

One of the great strengths of ``moments.LD`` is that while it only computes low-order
moments of the full two-locus haplotype distribution, it allows us to expand the basis
of statistics to include many populations:

.. jupyter-execute::

    y = moments.LD.Demographics2D.split_mig((0.5, 2.0, 0.2, 1.0), rho=1.0)
    print(y.names())
    y

Notice that already with just two populations we pick up many additional statistics:
not just :math:`\mathbb{E}[D_0^2]` and :math:`\mathbb{E}[D_1^2]`, but also the cross
population covariance of :math:`D`: :math:`\mathbb{E}[D_0 D_1]`, as well as all possible
combinations of :math:`D`, :math:`p`, and :math:`q` for the ``Dz`` and ``pi2`` moments.
This is what makes such LD computation an efficient and powerful approach for inference:
it is very fast to compute, it can be extended to many populations, and it gives us
a large set informative statistics to compare to data.

******************
Demographic events
******************

Mirroring the ``moments`` API for manipulating SFS, we can apply demographic
events to LD objects using demographic functions that return a *new* LDstats object:

Extinction/marginalization
--------------------------

If a population goes extinct, or if we just want to stop tracking statistics involving
that population, we can use ``y.marginalize(idx)`` to remove a given population or
set of populations from the LD stats. Here, ``idx`` can be either an integer index or
a list of integer indexes. ``y.marginalize()`` returns a new LD stats object with the
specified populations removed and the population IDs preserved for the remaining
populations (if given in the input LD stats).

Population splits
-----------------

To split one population, we use ``y.split(i, new_ids=["child1", "child2"])``, where
``i`` is the integer index of the population to split, and the optional argument
``new_ids`` lets us set the split population IDs. Note that if the input LD stats do
not have population IDs defined (i.e ``y.pop_ids == None``), we cannot specify new
IDs.

Admixture and mergers
---------------------

Admixture and merge events take two populations and combine them with given fractions
of ancestry from each. The new admixed/merged population is placed at the end of the
array of population indexes, and the only difference been ``y.admix()`` and
``y.merge()`` is that the ``merge`` function then removes the parental populations
(i.e. the parents are marginalized after admixture).

For both functions, usage is ``y.admix(idx0, idx1, f, new_id="xxx")``. We specify
the indexes of the two parental populations (``idx0`` and ``idx1``) and the proportion
``f`` contributed by the first specified population ``idx0`` (population ``idx1``
contributes 1-``f). We can also provide the ID of the admixed population using
``new_id``:

.. jupyter-execute::

    y = moments.LD.Demographics2D.snm(pop_ids=["A", "B"])
    print(y.pop_ids)
    y = y.admix(0, 1, 0.2, new_id="C")
    print(y.pop_ids)
    y = y.merge(1, 2, 0.75, new_id="D")
    print(y.pop_ids)

Pulse migration
---------------

Finally, we can apply discrete (or pulse) mass migration events with a given
proportion from one population to another. Here, we again specify 1) the index
of the source population, 2) the index of the target/destination population, and
3) the proportion of ancestry contributed:

.. jupyter-execute::

    y = y.pulse_migrate(1, 0, 0.1)
    print(y.pop_ids) # population IDs are unchanged.

***********
Integration
***********

Integrating the LD stats also mirrors the SFS integration function, with some changes
to keyword arguments. At a minimum, we need to specify the relative sizes or size
function ``nu`` and the integration time ``T``. When simulating LD stats for one or
more recombination rates, we also pass ``rho`` as a single rate or a list of rates,
as needed:

.. code-block::

    y.integrate(nu, T, rho=rho, theta=theta)

For multiple populations, we can also specify a migration matrix of size
:math:`n \times n`, where :math:`n` is the number of populations that the LD stats
represents. Like the SFS integration, we can also specify any populations that are
frozen by passing a list of length :math:`n` with ``True`` for frozen populations and
``False`` for populations to integrate.

Unlike SFS integration, LD integration also lets us specify selfing rates within each
population, where ``selfing`` is a list of length :math:`n` that specifies the selfing
rate within each deme, which must be between 0 and 1.

*********
Inference
*********

.. todo:: Still need to finish this section of the documentation.

*******
Parsing
*******

.. todo:: Still need to finish this section of the documentation.

**********
References
**********

.. [Hill1968]
    Hill and Robertson...
