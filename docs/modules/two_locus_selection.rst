.. _two-locus-usage:

=====================
Selection at two loci
=====================
.. jupyter-kernel:: python3

By Aaron Ragsdale, January 2021.

.. todo:: This module has not been completed

Most users of ``moments`` will be most interested in computing the single-site SFS and
comparing it to data. However, ``moments`` can do much more, such as computing expectations
for LD under complex demography, or triallelic or two-locus frequency spectra. Here, we'll
explore what we can do with the two-locus methods available in ``moments.TwoLocus``.

.. jupyter-execute::

    import moments.TwoLocus
    import numpy as np
    import matplotlib.pylab as plt

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
one or both loci. While ``moments.TwoLocus`` has a reversible mutation model implemented,
here we'll focus on the infinite sites model (ISM), under the assumption that
:math:`N_e \mu \ll 1` at both loci.

Below, we'll walk through how to compute the sampling distribution
for two-locus haplotypes for a given sample size, describe its relationship to common
measures of linkage disequilibrium (LD), and explore how recombination, demography, and
selection interacts to alter expected patterns of LD. In particular, we'll focus on
a few different models of selection, dominance, and epistasic interactions between loci,
and ask under what conditions those patterns are expected to differ or to be confounded.

Citing this work
++++++++++++++++

.. todo:: [Ragsdale_Gutenkunst]_, [Ragsdale_Gravel]_

Two-locus haplotype distribution under neutrality
=================================================

.. todo:: Some basics about working with the two-locus spectrum, comparing to ``moments.LD``
    how the size of the data array grows (and efficiency declines) with increasing sample
    size, caching results can help a lot.

.. todo:: Comparison to ``moments.LD`` for single population scenarios (equilibrium,
    bottleneck, growth). Recreate figure from Hudson 2002 of n_11 for different
    recombination rates and possible different demographic scenarios.

.. note:: We'll want to cache the equilibrium spectra, unless we can figure out how to
    get the equilibrium spectrum from :math:`\Psi^0=A^{-1}\cdot(-M_{0,1})` with :math:`A`
    not being singular. This would be a lot cleaner and help a ton with "burn-in".

How does selection interact across multiple loci?
=================================================

There has been a recent resurgence of interest in learning about the interaction of
selection at two or more loci (e.g., for studies within the past few years, see
[Sohail]_, [Garcia]_, [Sandler]_, [Good]_). This has largely been driven by the
relatively recent availability of large-scale sequencing datasets that allow us to
observe patterns of allele frequencies and LD for negatively selected loci that may
be segregating at very low frequencies in a population. Some of these studies are
theory-driven (e.g., [Good]_), while others rely on forward Wright-Fisher simulators
(such as ``SLiM`` or ``fwdpy11``) to compare observed patterns between data and
simulation.

These approaches have their limitations: analytical results are largely
constrained to simple selection scenarios and steady-state demography, while simulation
studies are computationally expensive and thus often end up limited to still a handful
of selection and demographic scenarios. Numerical approaches to compute expectations of
statistics of interest could therefore provide a far more efficient way to compute
explore parameter regimes and compare model expectations to data in inference frameworks.

Here, we'll explore a few selection models, including both dominance and epistatic
effects, that theory predicts should result in different patterns of LD between two
selected loci. We first describe the selection models, and then we compare their
expected patterns of LD.

Selection models at two loci
++++++++++++++++++++++++++++

At a single locus, the effects of selection and dominance are captured by the selection
coefficient :math:`s` and the dominance coefficient :math:`h`, so that fitnesses of the
diploid genotypes are given by

.. list-table:: Single-locus fitnesses.
    :align: center

    * - Genotype
      - Relative fitness
    * - `aa`
      - :math:`1`
    * - `Aa`
      - :math:`1 + 2hs`
    * - `AA`
      - :math:`1 + 2s`

If :math:`h = 1/2`, i.e. selection is `additive`, this model reduces to a haploid
selection model where genotype `A` has relative fitness :math:`1 + s` compared to `a`.

Additive selection, no epistasis
--------------------------------

Additive selection models for two loci, like in the single-locus case, reduce to
haploid-based models, where we only need to know the relative fitnesses of the two-locus
haplotypes `AB`, `Ab`, `aB`, and `ab`. When we say "no epistasis," we typically mean that
the relative fitness of an individual carrying both derived alleles (`AB`) is additive
across loci, so that if :math:`s_A` is the selection coefficient at the left (`A/a`)
locus, and :math:`s_B` is the selection coefficient at the right (`B`/`b`) locus, then
:math:`s_{AB} = s_A + s_B`.

.. list-table:: No epistasis or dominance emits a haploid selection model.
    :align: center

    * - Genotype
      - Relative fitness
    * - `ab`
      - :math:`1`
    * - `Ab`
      - :math:`1 + s_A`
    * - `aB`
      - :math:`1 + s_B`
    * - `AB`
      - :math:`1 + s_{AB} = 1 + s_A + s_B`

Additive selection with epistasis
---------------------------------

Epistasis is typically modeled as a factor :math:`\epsilon` that either increases or
decreases the selection coefficient for the `AB` haplotype, so that
:math:`s_{AB} = s_A + s_B + \epsilon`. If :math:`|s_{AB}| > |s_A| + |s_A|`, i.e. the
fitness effect of the `AB` haplotype is greater than the sum of the effect of the `Ab`
and `aB` haplotypes, the effect is called `synergistic` epistasis, and if
:math:`|s_{AB}| < |s_A| + |s_A|`, it is refered to as `antagonistic` epistasis.

.. list-table:: A haploid selection model with epistasis.
    :align: center

    * - Genotype
      - Relative fitness
    * - `ab`
      - :math:`1`
    * - `Ab`
      - :math:`1 + s_A`
    * - `aB`
      - :math:`1 + s_B`
    * - `AB`
      - :math:`1 + s_{AB} = 1 + s_A + s_B + \epsilon`

Simple dominance, no epistasis
------------------------------

Epistasis is the non-additive interaction of selective effects across loci. The
non-additive effect of selection within a locus is called dominance, when
:math:`s_{AA} \not= 2s_{Aa}`. Without epistasis, so that :math:`s_{AB}=s_{A}+s_{B}`,
and allowing for different selection and dominance coefficients at the two loci,
the fitness effects for two-locus diploid genotypes takes a simple form analogous
to the single-locus case with dominance. Here, we define the relative fitnesses of
two-locus diploid genotypes, which relies on the selection and dominance coefficients
at the left and right loci:

.. list-table:: Accounting for dominance requires modeling selection for diploid
    genotypes, instead of the model reducing to selection on haploid genotypes.
    :align: center

    * - Genotype
      - Relative fitness
    * - `aabb`
      - :math:`1`
    * - `Aabb`
      - :math:`1 + 2 h_A s_A`
    * - `AAbb`
      - :math:`1 + 2 s_A`
    * - `aaBb`
      - :math:`1 + 2 h_B s_B`
    * - `AaBb`
      - :math:`1 + 2 h_A s_A + 2 h_B s_B`
    * - `AABb`
      - :math:`1 + 2 s_A + 2 h_B s_B`
    * - `aaBB`
      - :math:`1 + 2 s_B`
    * - `AaBB`
      - :math:`1 + 2 h_A s_A + 2 s_B`
    * - `AABB`
      - :math:`1 + 2 s_A + 2 s_B`

Both dominance and epistasis
----------------------------

As additional non-additive interactions are introduced, it gets more difficult to
succinctly define general selection models with few parameters. A general selection
model that is flexible could simply define a selection coefficient for each two-locus
diploid genotype, in relation to the double wild-type homozygote (`aabb`). That is, define
:math:`s_{Aabb}` as the selection coefficient for the `Aabb` genotype, :math:`s_{AaBb}`
the selection coefficient for the `AaBb` genotype, and so on. 

Gene-based dominance
--------------------

In the above model, fitness is determined by combined hetero-/homozygosity at the two loci,
but it does not make a distinction between the different ways that double heterozygotes
(`AaBb`) could arise. Instead, we could imagine a model where diploid individual fitnesses
depend on the underlying haplotypes, i.e. whether selected mutations at the two loci are
coupled on the same background or are on different haplotypes.

For example, consider loss-of-function mutations in coding regions. Such mutations tend
to be severely damaging. We could think of the situation where diploid individual fitness
is strongly reduced when both copies carry a loss-of-function mutation, but much less
reduced if the individual has at least one copy without a mutation. In this scenario,
the haplotype combination `Ab / aB` will confer more reduced fitness compared to the
combination `AB / ab`, even though both are double heterozygote genotypes. 

Perhaps the simplest model for gene-based dominance assumes that derived mutations at
the two loci (`A` and `B`) carry the same fitness cost, and fitness depends on the number
of haplotype copies within a diploid individual that have at least one such mutation. This
model requires just two parameters, a single selection coefficient `s` and a single
dominance coefficient `h`:

.. list-table:: A simple gene-based dominance model.
    :align: center

    * - Genotype
      - Relative fitness
    * - `ab / ab`
      - :math:`1`
    * - `Ab / ab`
      - :math:`1 + 2 h s`
    * - `aB / ab`
      - :math:`1 + 2 h s`
    * - `AB / ab`
      - :math:`1 + 2 h s`
    * - `Ab / Ab`
      - :math:`1 + 2 s`
    * - `aB / aB`
      - :math:`1 + 2 s`
    * - `Ab / aB`
      - :math:`1 + 2 s`
    * - `AB / Ab`
      - :math:`1 + 2 s`
    * - `AB / aB`
      - :math:`1 + 2 s`
    * - `AB / AB`
      - :math:`1 + 2 s`

.. note:: Cite [Sanjak]_

How do the selection models affect expected LD statistics?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. todo:: All the comparisons, show LD curves and expectations for signed LD, depending
    on the selection model, maybe explore how population size changes distort these
    expectations.

.. todo:: Discussion on what we can expect to learn from signed LD-based inferences. Are
    the various selection models and demography hopelessly confounded?

References
==========

.. [Garcia]
    Garcia, Jesse A., and Kirk E. Lohmueller. "Negative linkage disequilibrium between
    amino acid changing variants reveals interference among deleterious mutations in the
    human genome." *bioRxiv* (2020).

.. [Good]
    Good, Benjamin H. "Linkage disequilibrium between rare mutations." bioRxiv (2020).

.. [Ragsdale_Gutenkunst]
    Ragsdale, Aaron P. and Ryan N. Gutenkunst. "Inferring demographic history using
    two-locus statistics." *Genetics* 206.2 (2017): 1037-1048.

.. [Ragsdale_Gravel]
    Ragsdale, Aaron P. and Simon Gravel. "Models of archaic admixture and recent history
    from two-locus statistics." *PLoS Genetics* 15.8 (2019): e1008204.

.. [Sandler]
    Sandler, George, Stephen I. Wright, and Aneil F. Agrawal. "Using patterns of signed
    linkage disequilibria to test for epistasis in flies and plants." *bioRxiv* (2020).

.. [Sanjak]
    is a sweet paper

.. [Sohail]
    Sohail, Mashaal, et al. "Negative selection in humans and fruit flies involves
    synergistic epistasis." *Science* 356.6337 (2017): 539-542.
