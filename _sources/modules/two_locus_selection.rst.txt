.. _two-locus-usage:

=====================
Selection at two loci
=====================
.. jupyter-kernel:: python3

By Aaron Ragsdale, January 2021.

.. note:: This module has not been completed - I've placed to-dos where content is
    incoming. If you find an error here, or find some aspects confusing, please don't
    hesitate to get in touch or open an issue. Thanks!

Most users of ``moments`` will be most interested in computing the single-site SFS and
comparing it to data. However, ``moments`` can do much more, such as computing expectations
for LD under complex demography, or triallelic or two-locus frequency spectra. Here, we'll
explore what we can do with the two-locus methods available in ``moments.TwoLocus``.

.. jupyter-execute::

    import moments.TwoLocus
    import numpy as np
    import matplotlib.pylab as plt
    import pickle, gzip

.. jupyter-execute::
    :hide-code:

    import matplotlib
    plt.rcParams['legend.title_fontsize'] = 'xx-small'
    matplotlib.rc('xtick', labelsize=9)
    matplotlib.rc('ytick', labelsize=9)
    matplotlib.rc('axes', labelsize=12)
    matplotlib.rc('axes', titlesize=12)
    matplotlib.rc('legend', fontsize=10)

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

Demographic inference using a diffusion approximation-based solution for :math:`\Psi_n`
was introduced in [Ragsdale_Gutenkunst]_. The moments-based method, which is implemented
here, was described in [Ragsdale_Gravel]_, and the selection models were described in
[Ragsdale2022]_. If you use ``moments.TwoLocus`` in your research, please cite
[Ragsdale2022]_.

Two-locus haplotype distribution under neutrality
=================================================

A quick comment on computational efficiency
+++++++++++++++++++++++++++++++++++++++++++

The frequency spectrum :math:`\Psi_n` is displayed as a 3-dimensional array in `moments`,
and the size grows quite quickly in the sample size :math:`n`. (The number of frequency
bins is :math:`\frac{1}{6}(n+1)(n+2)(n+3)`, so it grows as :math:`n^3`.) Thus, solving
for :math:`\Psi` gets quite expensive for large sample sizes.

.. jupyter-execute::
    :hide-code:

    from matplotlib.ticker import ScalarFormatter
    ns = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
    t_jk = np.array(
        [1.01, 1.48, 3.03, 7.65, 20.58, 47.07, 102.44, 412.94, 1192.44, 3526.23]
    )
    mem_jk = np.array(
        [131828, 135580, 141608, 157068, 187996, 253480, 345692, 700044, 1601448, 3543684]
    )
    t_no_jk = np.array(
        [0.87, 0.93, 1.02, 1.43, 2.21, 4.24, 8.31, 26.66, 77.12, 280.66]
    )
    mem_no_jk = np.array(
        [130864, 133832, 140784, 156116, 185932, 247812, 341120, 692348, 1598728, 3541400]
    )

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(ns, t_jk, label="With jackknife computation")
    ax1.plot(ns, t_no_jk, label="Cached jackknife")
    ax1.set_xlabel("Sample size")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.xaxis.set_minor_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax1.set_title("Time to compute equilibrium FS")

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(ns, mem_jk / 1024, label="With jackknife computation")
    ax2.plot(ns, mem_no_jk / 1024, label="Cached jackknife")
    ax2.set_xlabel("Sample size")
    ax2.set_ylabel("Mb")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.xaxis.set_minor_formatter(ScalarFormatter())
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_title("Maximum memory usage")
    fig.tight_layout()

Here, we see the time needed to compute the equilibrium frequency spectrum for a given
sample size. Recombination requires computing a jackknife operator for approximate
moment closure, which gets expensive for large sample sizes. However, we can
cache and reuse this jackknife matrix (the default behavior), so that much of the
computational time is saved from having to recompute that large matrix. However, we see
that simply computing the steady-state solution still
gets quite expensive as the sample sizes increase.

Below, we'll see that for
non-zero recombination (as well as selection) our accuracy improves as we increase the
sample size. For this reason, we've pre-computed and cached results throughout this
page, and the code blocks give examples of how those results were created.

Two neutral loci
++++++++++++++++

The ``moments.TwoLocus`` solution for the neutral frequency spectrum without recombination
(:math:`\rho = 4 N_e r = 0`) is exact, while :math:`\rho > 0` and selection require a
moment-closure approximation. This approximation grows more accurate for larger :math:`n`.

To get familiar with some common two-locus statistics (either summaries of :math:`\Psi_n`
and :math:`\Psi` itself), we can compare to some classical results, such as the expectation
for :math:`\sigma_d^2 = \frac{\mathbb{E}[D^2]}{\mathbb{E}[p(1-p)q(1-q)]}`, where `D` is
the standard covariance measure of LD, and `p` and `q` are allele frequencies at the
left and right loci, respectively [Ohta]_:

.. jupyter-execute::

    rho = 0
    n = 10
    Psi = moments.TwoLocus.Demographics.equilibrium(n, rho=rho)
    sigma_d2 = Psi.D2() / Psi.pi2()
    print(r"moments.TwoLocus $\sigma_d^2$, $r=0$:", sigma_d2)
    print(r"Ohta and Kimura expectation, $r=0$:", 5 / 11)

And we can plot the LD-decay curve for :math:`\sigma_d^2` for a range of recombination
rates and a few different sample sizes, and compare to [Ohta]_'s expectation, which is
:math:`\sigma_d^2 = \frac{5 + \frac{1}{2}\rho}{11 + \frac{13}{2}\rho + \frac{1}{2}\rho^2}`:

.. jupyter-execute::
    :hide-code:

    rhos_ok = np.logspace(-1, 2, 50)
    ohta_kimura = (5 + rhos_ok / 2) / (11 + 13 * rhos_ok / 2 + rhos_ok ** 2 / 2)
    rhos = np.logspace(-1, 2, 11)
    ok_compare = (5 + rhos / 2) / (11 + 13 * rhos / 2 + rhos ** 2 / 2)
    # precomputed using `F = moments.TwoLocus.Demographics.equilibrium(n, rho=rho)`
    # and then `F.D2() / F.pi2()`
    ld_curve_moments = {
        20: [0.4332, 0.4138, 0.3799, 0.3264, 0.2547, 0.1774, 0.1108, 0.0634, 0.0339, 0.0172, 0.0045],
        30: [0.4332, 0.4139, 0.3801, 0.3269, 0.2556, 0.1786, 0.1121, 0.0646, 0.035, 0.0248, 0.0074],
        50: [0.4333, 0.414, 0.3803, 0.3272, 0.2562, 0.1794, 0.1128, 0.0652, 0.0356, 0.0186, 0.2883],
        80: [0.4333, 0.414, 0.3803, 0.3273, 0.2565, 0.1797, 0.1131, 0.0655, 0.0357, -0.0117, -0.6302],
    }

    fig = plt.figure(figsize=(6, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(rhos_ok, ohta_kimura, 'k--', lw=2, label="Ohta and Kimura")
    for n in sorted(ld_curve_moments.keys()):
        ax1.plot(rhos, ld_curve_moments[n], "v-", lw=1, label=f"moments, n={n}")
    ax1.set_ylabel(r"$\sigma_d^2$")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(rhos_ok, rhos_ok * 0, "k--", lw=2, label=None)
    for n in sorted(ld_curve_moments.keys()):
        ax2.plot(
            rhos[:-2],
            ((ld_curve_moments[n] - ok_compare) / ok_compare)[:-2],
            "v-",
            lw=1,
            label=f"moments, n={n}"
        )
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylabel("Percent error")
    ax2.set_xlabel(r"$\rho$")
    ax2.set_xscale("log")
    ax2.legend()
    
    fig.tight_layout()


The moments approximation breaks down for recombination rates around :math:`\rho\approx50`
but is very accurate for lower recombination rates, and this accuracy increases with the
sample size. To be safe, we can assume that numerical error starts to creep
in around :math:`\rho\approx25`, which for human parameters, is very roughly 50 or 100kb.
So we're limited to looking at LD in relatively shorter regions. For higher recombination
rates, we can turn to ``moments.LD``, which lets us model multiple populations, but
is restricted to neutral loci and low-order statistics.

The statistics :math:`\mathbb{E}[D^2]` and :math:`\mathbb{E}[p(1-p)q(1-q)]` are low-order
summaries of the full sampling distribution, similar to how heterozygosity or Tajima's `D`
are low-order summaries of the single-site SFS. We can visualize some features of the
full two-locus haplotype frequency distribution instead, following Figure 1 in Hudson's
classical paper on the two-locus sampling distribution [Hudson]_. Here, we'll look at
a slice in the 3-dimensional distribution: if we observe :math:`n_A` samples carrying `A`
at the left locus, and :math:`n_B` carrying `B` at the right locus, what is the probability
that we observe `n_{AB}` haplotypes with `A` and `B` coupled in the same sample? This
marginal distribution will depend on :math:`\rho`:

.. jupyter-execute::

    rhos = [0.5, 5.0, 30.0]
    n = 30
    nA = 15
    nB = 12

    # first we'll get the slice for the given frequencies from the "hnrho" file
    # from RR Hudson: http://home.uchicago.edu/~rhudson1/source/twolocus.html
    hudson = {}
    import gzip
    with gzip.open("./data/h30rho.gz", "rb") as fin:
        at_frequencies = False
        for line in fin:
            l = line.decode()
            if "freq" in l:
                if int(l.split()[1]) == nA and int(l.split()[2]) == nB:
                    at_frequencies = True
                else:
                    at_frequencies = False
            if at_frequencies:
                rho = float(l.split()[1])
                if rho in rhos:
                        hudson[rho] = np.array([float(v) for v in l.split()[2:]])

    fig = plt.figure(figsize=(12, 4))
    for ii, rho in enumerate(rhos):
        # results are cached, having used the following line to create the spectra
        # F = moments.TwoLocus.Demographics.equilibrium(n, rho=rho)
        F = pickle.load(gzip.open(f"./data/two-locus/eq.n_{n}.rho_{rho}.fs.gz", "rb"))
        counts, pAB = moments.TwoLocus.Util.pAB(F, nA, nB)
        pAB /= pAB.sum()
        ax = plt.subplot(1, 3, ii + 1)
        ax.bar(counts - 0.2, hudson[rho] / hudson[rho].sum(), width=0.35, label="Hudson")
        ax.bar(counts + 0.2, pAB, width=0.35, label="moments.TwoLocus")
        ax.set_title(f"rho = {rho}")
        if ii == 0:
            ax.set_ylabel("Probability")
            ax.legend()
        if ii == 1:
            ax.set_xlabel(r"$n_{AB}$")
    fig.tight_layout()

For low recombination rates, the marginal distribution of `AB` haplotypes is skewed
toward the maximum or minimum number of copies, resulting in higher LD, while for larger
recombination rates, the distribution of :math:`n_{AB}` is concentrated around frequencies
that result in low levels of LD. We can also see that ``moments.TwoLocus`` agrees well
with Hudson's results under neutrality and steady state demography.

.. note:: Below, we'll be revisiting these same statistics and seeing how various models
    of selection at the two loci, as well as non-steady state demography, distort the
    expected distributions.

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
statistics of interest could therefore provide a far more efficient way to
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

.. list-table:: No epistasis or dominance yields a haploid selection model.
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
:math:`s_{AB} = s_A + s_B + \epsilon`. If :math:`|s_{AB}| > |s_A| + |s_B|`, i.e. the
fitness effect of the `AB` haplotype is greater than the sum of the effect of the `Ab`
and `aB` haplotypes, the effect is called `synergistic` epistasis, and if
:math:`|s_{AB}| < |s_A| + |s_B|`, it is referred to as `antagonistic` epistasis.

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

How do different selection models affect expected LD statistics?
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here, we will examine some relatively simple models in order to gain some intuition about
how selection, along with recombination and size changes, affect expected patterns of LD,
such as the decay curve of :math:`\sigma_d^2` and Hudson-style slices in the two-locus
sampling distribution. The selection coefficients will be equal at the two loci, so that
the only selection parameters that change will be the selection models (dominance and
epistasis).

Additive selection with and without epistasis
---------------------------------------------

Let's first see how simple, additive selection distorts expected LD away from neutral
expectations at steady state. Plotted below are decay curves for both :math:`\sigma_d^2`
and :math:`\sigma_d^1 = \frac{\mathbb{E}[D]}{\mathbb{E}[p(1-p)q(1-q)]}`, a common signed LD
statistic.

For each parameter pair of selection coefficient :math:`\gamma = 2 N_e s` and :math:`\rho`,
we use the "helper" function that creates the input selection parameters for the
`AB`, `Ab`, and `aB` haplotypes, and then simulate the equilibrium two-locus sampling
distribution:

.. code-block:: python
    
    sel_params = moments.TwoLocus.Util.additive_epistasis(gamma, epsilon=0)
    # epsilon=0 means no epistasis, so s_AB = s_A + s_B
    F = moments.TwoLocus.Demographics.equilibrium(n, rho=rho, sel_params=sel_params)
    sigma_d1 = F.D() / F.pi2()
    sigma_d2 = F.D2() / F.pi2()

.. note:: Additive selection with the same coefficient at both loci and no epistasis
    can also be specified with the ``gamma`` keyword argument, e.g.
    ``moments.TwoLocus.Demographics.equilibrium(n, rho=rho, gamma=gamma)``, which is
    shorthand for ``sel_params=[2 * gamma, gamma, gamma]``. This is selection at
    `both` loci. For selection at only the left locus, use
    ``sel_params=[gamma, gamma, 0]``.

.. jupyter-execute::
    :hide-code:

    rhos = np.concatenate((np.logspace(-1, 1, 20), [15, 20, 30]))
    # signed $D$, normalized by $E[p(1-p)q(1-q)]$, i.e. F.D() / F.pi2()
    sigma_d1 = {
        -0.1: [-0.003204, -0.003126, -0.003034, -0.002926, -0.002808, -0.002676, -0.002536, -0.002392, -0.002248, -0.002106, -0.001968, -0.00183, -0.001682, -0.001522, -0.001342, -0.001146, -0.000938, -0.000734, -0.000544, -0.000378, -0.000172, -8.6e-05, -4.6e-05],
        -1.0: [-0.179838, -0.175396, -0.169966, -0.163396, -0.155554, -0.146336, -0.135706, -0.123728, -0.110584, -0.096604, -0.082244, -0.068062, -0.054636, -0.042478, -0.031968, -0.023294, -0.016454, -0.01129, -0.007552, -0.004944, -0.002362, -0.0014, -0.000716],
        -5.0: [-0.063176, -0.062636, -0.061974, -0.061168, -0.060196, -0.059034, -0.05766, -0.056054, -0.054192, -0.052054, -0.04962, -0.04688, -0.043822, -0.04045, -0.036792, -0.032898, -0.02886, -0.0248, -0.020868, -0.017224, -0.012138, -0.009472, -0.00705],
        -20.0: [0.006202, 0.0062, 0.006198, 0.006196, 0.006192, 0.006186, 0.00618, 0.006172, 0.00616, 0.006148, 0.00613, 0.00611, 0.006082, 0.006048, 0.006004, 0.00595, 0.00588, 0.005794, 0.005684, 0.005548, 0.005238, 0.004936, 0.004354]
    }
    # classical $\sigma_d^2$ statistic
    sigma_d2 = {
        -0.1: [0.424871, 0.419505, 0.412863, 0.404701, 0.39476, 0.382787, 0.368553, 0.351898, 0.332764, 0.31124, 0.287596, 0.262286, 0.235934, 0.20927, 0.183053, 0.157986, 0.13464, 0.113417, 0.09454, 0.078068, 0.055661, 0.043278, 0.029954],
        -1.0: [0.321948, 0.318345, 0.31388, 0.308387, 0.301688, 0.293603, 0.283965, 0.272646, 0.259577, 0.244779, 0.228384, 0.210641, 0.191916, 0.172658, 0.153359, 0.134505, 0.11653, 0.09978, 0.0845, 0.070828, 0.051639, 0.040697, 0.028609],
        -5.0: [0.061941, 0.06176, 0.061537, 0.061261, 0.060922, 0.060509, 0.060007, 0.059402, 0.058677, 0.05781, 0.056778, 0.055555, 0.054113, 0.052422, 0.050454, 0.048189, 0.045616, 0.042742, 0.039592, 0.036216, 0.030266, 0.026015, 0.020318],
        -20.0: [0.012609, 0.012604, 0.012598, 0.01259, 0.01258, 0.012567, 0.01255, 0.012529, 0.012503, 0.012469, 0.012426, 0.012372, 0.012304, 0.012218, 0.01211, 0.011976, 0.011808, 0.011602, 0.011348, 0.011041, 0.010385, 0.009800, 0.008803]
    }

    fig = plt.figure(figsize=(6, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(rhos_ok[:-5], ohta_kimura[:-5], 'k--', lw=2, label="Ohta & Kimura")
    for gamma in sorted(sigma_d2.keys())[::-1]:
        ax1.plot(rhos, sigma_d2[gamma], "v-", lw=1, label=rf"$\gamma = {gamma}$")
    ax1.set_ylabel(r"$\sigma_d^2$")
    ax1.set_xlabel(r"$\rho$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    fig.tight_layout()

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(rhos_ok[:-5], 0 * ohta_kimura[:-5], 'k--', lw=2, label="Neutrality")
    for gamma in sorted(sigma_d1.keys())[::-1]:
        ax2.plot(rhos, sigma_d1[gamma], "v-", lw=1, label=rf"$\gamma = {gamma}$")
    ax2.set_ylabel(r"$\sigma_d^1$")
    ax2.set_xlabel(r"$\rho$")
    ax2.set_xscale("log")
    ax2.legend()
    fig.tight_layout()

Already with this very simple selection model (no epistasis, no dominance, equal selection
at both loci), we find some interesting behavior. For very strong or very weak selection,
signed-LD remains close to zero, but for intermediate selection, average :math:`D` can be
significantly negative. As fitness effects get stronger, :math:`\sigma_d^2` is reduced
dramatically compared to neutral expectations.

.. todo:: Plots of frequency conditioned LD.

The "helper" function that we used above converts input :math:`\gamma` and :math:`\epsilon`
to the selection parameters that are passed to ``moments.TwoLocus.Demographics`` functions.
The additive epistasis model implemented in the helper function
(``moments.TwoLocus.Util.additive_epistasis``) uses the same selection coefficient
at both loci and returns :math:`[2(1+\epsilon)\gamma, \gamma, \gamma]`, so that if
:math:`\epsilon > 0`, we have synergistic epistasis, and if :math:`\epsilon < 0`, we
have antagonistic epistasis. Any value of :math:`\epsilon` is permitted, and note that if
:math:`\epsilon` is less than :math:`-1`, we get reverse-sign epistasis.

We'll focus on two selection regions: mutations that are slightly deleterious with
:math:`\gamma=1`, and stronger selection with :math:`\gamma=20`. With an effective
population size of 10,000, note that :math:`\gamma=20` corresponds to
:math:`s=0.001` - by no means a lethal mutation, but strong enough to see some interesting
differences between selection regimes.

Below we again plot :math:`\sigma_d^2` and :math:`\sigma_d^1` for each set of parameters:

.. jupyter-execute::

    gammas = [-1, -20]
    epsilons = [-1, -0.5, 0, 0.5, 1]

.. jupyter-execute::
    :hide-code:
    
    rhos = np.logspace(-1, np.log10(30), 15)
    sigma_d2s = {
        -1: {
            -1.0: [0.448777, 0.439449, 0.426149, 0.407616, 0.382597, 0.350211, 0.310498, 0.264945, 0.216624, 0.169551, 0.127428, 0.092531, 0.065403, 0.045313, 0.030942],
            -0.5: [0.382815, 0.374704, 0.363176, 0.347191, 0.325761, 0.298282, 0.264964, 0.227183, 0.187431, 0.14874, 0.113804, 0.084313, 0.060792, 0.042875, 0.02971],
            0: [0.321948, 0.315408, 0.306112, 0.293224, 0.275948, 0.253795, 0.226916, 0.196357, 0.164002, 0.132145, 0.102871, 0.077596, 0.05692, 0.040764, 0.028609],
            0.5: [0.281531, 0.276047, 0.268241, 0.257398, 0.242832, 0.224101, 0.201302, 0.17528, 0.147587, 0.12012, 0.094617, 0.072289, 0.053717, 0.038938, 0.02762],
            1.0: [0.25898, 0.254014, 0.246939, 0.237097, 0.223851, 0.206786, 0.185982, 0.162218, 0.136935, 0.111878, 0.088612, 0.068189, 0.051098, 0.037369, 0.026733],
        },   
        -20: {
            -1.0: [0.869571, 0.866524, 0.861953, 0.855107, 0.84487, 0.829602, 0.806933, 0.773515, 0.724854, 0.655516, 0.560562, 0.439787, 0.305373, 0.184427, 0.100443],
            -0.5: [0.047087, 0.046997, 0.046861, 0.046659, 0.046357, 0.04591, 0.045253, 0.044297, 0.042926, 0.041004, 0.038389, 0.034981, 0.03078, 0.025953, 0.020852],
            0: [0.012609, 0.0126, 0.012587, 0.012566, 0.012536, 0.01249, 0.012422, 0.012321, 0.012173, 0.011956, 0.011644, 0.011205, 0.010601, 0.009805, 0.008803],
            0.5: [0.005881, 0.005879, 0.005876, 0.005872, 0.005865, 0.005856, 0.005841, 0.00582, 0.005788, 0.005741, 0.00567, 0.005567, 0.005417, 0.005202, 0.004905],
            1.0: [0.003518, 0.003518, 0.003517, 0.003516, 0.003514, 0.003511, 0.003507, 0.003501, 0.003492, 0.003478, 0.003457, 0.003425, 0.003377, 0.003303, 0.003193],
        }
    }
    sigma_d1s = {
        -1: {
            -1.0: [1.05236, 1.04511, 1.03352, 1.01495, 0.985592, 0.940624, 0.875342, 0.787458, 0.679682, 0.560486, 0.441646, 0.333946, 0.243938, 0.173404, 0.120746],
            -0.5: [0.435798, 0.434856, 0.43283, 0.428686, 0.420736, 0.40659, 0.383634, 0.35015, 0.306704, 0.25668, 0.205246, 0.15741, 0.116484, 0.083724, 0.058808],
            0: [-0.179838, -0.171816, -0.160712, -0.14591, -0.127166, -0.105006, -0.081018, -0.057676, -0.037558, -0.022316, -0.012158, -0.00615, -0.002954, -0.001404, -0.000716],
            0.5: [-0.658084, -0.64389, -0.623746, -0.59588, -0.558672, -0.51125, -0.454244, -0.390284, -0.323714, -0.25943, -0.201454, -0.15212, -0.112082, -0.080836, -0.057256],
            1.0: [-0.991936, -0.974736, -0.950132, -0.915706, -0.868996, -0.808138, -0.732824, -0.645168, -0.549858, -0.453212, -0.361534, -0.279648, -0.210272, -0.154146, -0.110542],
        },
        -20: {
            -1.0: [5.10762, 5.10301, 5.09606, 5.08559, 5.06977, 5.04582, 5.00942, 4.95376, 4.86785, 4.73367, 4.52151, 4.18634, 3.68033, 3.00751, 2.27613],
            -0.5: [1.01108, 1.00982, 1.00792, 1.00508, 1.00084, 0.994534, 0.985208, 0.971518, 0.951646, 0.923264, 0.883658, 0.83013, 0.760824, 0.675914, 0.578648],
            0: [0.006202, 0.0062, 0.006194, 0.006186, 0.006174, 0.006156, 0.006128, 0.006088, 0.00603, 0.005942, 0.005812, 0.005622, 0.005342, 0.004938, 0.004354],
            0.5: [-0.343972, -0.34382, -0.343592, -0.34325, -0.342738, -0.34197, -0.340824, -0.339114, -0.336576, -0.332836, -0.32737, -0.31949, -0.308352, -0.293026, -0.272718],
            1.0: [-0.520422, -0.520248, -0.51999, -0.519602, -0.519018, -0.518144, -0.516836, -0.514882, -0.511974, -0.507668, -0.50133, -0.492102, -0.478866, -0.46028, -0.434956],
        }
    }

    fig = plt.figure(figsize=(6, 8))
    markers = ["x", "+", ".", "v", "^"]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(rhos_ok[:-5], ohta_kimura[:-5], "k--", label="Ohta-Kimura")
    for ii, eps in enumerate(epsilons):
        ax1.plot(rhos, sigma_d2s[gammas[0]][eps],
            markers[ii] + "--", label=f"$\epsilon = {eps}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\sigma_d^2$")
    ax1.set_xlabel(r"$\rho$")
    ax1.legend()
    ax1.set_title(rf"$\gamma = {gammas[0]}$")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(rhos_ok[:-5], ohta_kimura[:-5], "k--", label="Ohta-Kimura")
    for ii, eps in enumerate(epsilons):
        ax2.plot(rhos, sigma_d2s[gammas[1]][eps],
            markers[ii] + "--", label=f"$\epsilon = {eps}$")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\rho$")
    ax2.legend()
    ax2.set_title(rf"$\gamma = {gammas[1]}$")
    fig.tight_layout()

From this, we can see that synergistic epistasis decreases :math:`\sigma_d^2` and
antagonistic epistasis increases it above expectations for :math:`\epsilon=0`. For signed
LD, however, both positive and negative :math:`\epsilon` push :math:`\sigma_d^1` farther
away from zero:

.. jupyter-execute::
    :hide-code:

    fig = plt.figure(figsize=(6, 8))
    markers = ["x", "+", ".", "v", "^"]
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(rhos_ok[:-5], 0 * ohta_kimura[:-5], "k--", label=None)
    for ii, eps in enumerate(epsilons):
        ax1.plot(rhos, sigma_d1s[gammas[0]][eps],
            markers[ii] + "--", label=f"$\epsilon = {eps}$")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\sigma_d^1$")
    ax1.set_xlabel(r"$\rho$")
    ax1.legend()
    ax1.set_title(rf"$\gamma = {gammas[0]}$")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(rhos_ok[:-5], 0 * ohta_kimura[:-5], "k--", label=None)
    for ii, eps in enumerate(epsilons):
        ax2.plot(rhos, sigma_d1s[gammas[1]][eps],
            markers[ii] + "--", label=f"$\epsilon = {eps}$")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\rho$")
    ax2.legend()
    ax2.set_title(rf"$\gamma = {gammas[1]}$")
    fig.tight_layout()

As expected, negative :math:`\epsilon` (i.e. selection against the `AB` haplotype is less
strong than the sum of selection against `A` and `B`) leads to an excess of coupling
LD (pairs with more `AB` and `ab` haplotypes) than repulsion LD (pairs with more `Ab`
and `aB` haplotypes).

We can see this effect more clearly by looking at a slice in the two-locus sampling
distribution. Since we're considering negative selection, we'll look at entries in the
sampling distribution with low frequencies at the two loci. For doubletons at both sites:

.. jupyter-execute::

    rhos = [0.5, 5.0, 30.0]
    n = 30
    nA = 2
    nB = 2

    epsilon = [-0.5, 0, 1]

    fig = plt.figure(figsize=(9, 3))
    for ii, rho in enumerate(rhos):
        pABs = {}
        for eps in epsilon:
            sel_params = moments.TwoLocus.Util.additive_epistasis(gammas[0], epsilon=eps)
            # F = moments.TwoLocus.Demographics.equilibrium(
            #     n, rho=rho, sel_params=sel_params)
            F = pickle.load(gzip.open(
                f"./data/two-locus/eq.n_{n}.rho_{rho}.sel_"
                + "_".join([str(s) for s in sel_params])
                + ".fs.gz",
                "rb"))
            counts, pAB = moments.TwoLocus.Util.pAB(F, nA, nB)
            pABs[eps] = pAB / pAB.sum()
        ax = plt.subplot(1, 3, ii + 1)
        ax.bar(counts - 0.25, pABs[epsilon[0]], width=0.22, label=rf"$\epsilon={epsilon[0]}$")
        ax.bar(counts, pABs[epsilon[1]], width=0.22, label=rf"$\epsilon={epsilon[1]}$")
        ax.bar(counts + 0.25, pABs[epsilon[2]], width=0.22, label=rf"$\epsilon={epsilon[2]}$")

        ax.set_title(rf"$\rho = {rho}$, $\gamma = {gammas[0]}$")
        ax.set_xlabel(r"$n_{AB}$")
        if ii == 0:
            ax.legend()
            ax.set_ylabel("Probability")
    fig.tight_layout()

.. jupyter-execute::

    fig = plt.figure(figsize=(9, 3))
    for ii, rho in enumerate(rhos):
        pABs = {}
        for eps in epsilon:
            sel_params = moments.TwoLocus.Util.additive_epistasis(gammas[1], epsilon=eps)
            # F = moments.TwoLocus.Demographics.equilibrium(
            #     n, rho=rho, sel_params=sel_params)
            F = pickle.load(gzip.open(
                f"./data/two-locus/eq.n_{n}.rho_{rho}.sel_"
                + "_".join([str(s) for s in sel_params])
                + ".fs.gz",
                "rb"))
            counts, pAB = moments.TwoLocus.Util.pAB(F, nA, nB)
            pABs[eps] = pAB / pAB.sum()
        ax = plt.subplot(1, 3, ii + 1)
        ax.bar(counts - 0.25, pABs[epsilon[0]], width=0.22, label=rf"$\epsilon={epsilon[0]}$")
        ax.bar(counts, pABs[epsilon[1]], width=0.22, label=rf"$\epsilon={epsilon[1]}$")
        ax.bar(counts + 0.25, pABs[epsilon[2]], width=0.22, label=rf"$\epsilon={epsilon[2]}$")

        ax.set_title(rf"$\rho = {rho}$, $\gamma = {gammas[1]}$")
        ax.set_xlabel(r"$n_{AB}$")
        if ii == 0:
            ax.legend()
            ax.set_ylabel("Probability")
    fig.tight_layout()

And while very few mutations will reach high frequency, we can also look at the case with
:math:`n_A=15` and :math:`n_B=12` in a sample size of 30. Here, because selection
and recombination require the jackknife approximation which works better with larger
sample sizes, we solved for the equilibrium distribution using size :math:`n=60` and then
projected to size 30.

.. code-block:: python

    n = 60
    n_proj = 30
    nA = 15
    nB = 12
    rho = 1

    F = moments.TwoLocus.Demographics.equilibrium(n, rho=rho, sel_params=sel_params)
    # by default, we usually cache projection steps, but set cache=False here to
    # save on memory usage
    F_proj = F.project(n_proj, cache=False)
    counts, pAB = moments.TwoLocus.Util.pAB(F_proj, nA, nB)
    pAB /= pAB.sum()

.. jupyter-execute:: 
    :hide-code:

    nABs = {
        -1: {
            -0.5: [0.338896, 0.049202, 0.02529, 0.016987, 0.013184, 0.011401, 0.01087, 0.011435, 0.013377, 0.017688, 0.027484, 0.056669, 0.407518],
            0: [0.452627, 0.063571, 0.031415, 0.020186, 0.014904, 0.012189, 0.010933, 0.01078, 0.011806, 0.014648, 0.021462, 0.042038, 0.293441],
            1: [0.656409, 0.085754, 0.038962, 0.022792, 0.015149, 0.011014, 0.008668, 0.007417, 0.007008, 0.007522, 0.009655, 0.016938, 0.112714],
        },
        -20: {
            -0.5: [1e-06, 0.0, 0.0, 1e-06, 2e-06, 5e-06, 1.5e-05, 4.6e-05, 0.000169, 0.000725, 0.003823, 0.028432, 0.966781],
            0: [0.000499, 7.8e-05, 6.8e-05, 8.5e-05, 0.000121, 0.000184, 0.000291, 0.000482, 0.00084, 0.001623, 0.004088, 0.022119, 0.969522],
            1: [0.414902, 0.040054, 0.016002, 0.010169, 0.007332, 0.005655, 0.004639, 0.004051, 0.003724, 0.003566, 0.003285, 0.008325, 0.478294],
        }
    }

    hudson_rho1 = [0.422518, 0.060438, 0.029775, 0.018964, 0.014617, 0.011862, 0.011203, 0.010706, 0.012594, 0.015119, 0.023559, 0.046696, 0.321949]

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 1, 1)
    ax.bar(np.arange(13) - 0.3, hudson_rho1, width=0.18, label="Hudson (neutrality)")
    ax.bar(np.arange(13) - 0.1, nABs[-1][-0.5], width=0.18, label=r"$\gamma=-1$, $\epsilon=-0.5$")
    ax.bar(np.arange(13) + 0.1, nABs[-1][0], width=0.18, label=r"$\gamma=-1$, $\epsilon=0$")
    ax.bar(np.arange(13) + 0.3, nABs[-1][1], width=0.18, label=r"$\gamma=-1$, $\epsilon=1$")
    ax.legend()
    ax.set_xlabel(r"$n_{AB}$")
    ax.set_ylabel("Probability")
    plt.suptitle("Weak selection with epistasis")
    fig.tight_layout()

.. jupyter-execute::
    :hide-code:

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 1, 1)
    ax.bar(np.arange(13) - 0.3, hudson_rho1, width=0.18, label="Hudson (neutrality)")
    ax.bar(np.arange(13) - 0.1, nABs[-20][-0.5], width=0.18, label=r"$\gamma=-20$, $\epsilon=-0.5$")
    ax.bar(np.arange(13) + 0.1, nABs[-20][0], width=0.18, label=r"$\gamma=-20$, $\epsilon=0$")
    ax.bar(np.arange(13) + 0.3, nABs[-20][1], width=0.18, label=r"$\gamma=-20$, $\epsilon=1$")
    ax.legend()
    ax.set_xlabel(r"$n_{AB}$")
    ax.set_ylabel("Probability")
    plt.suptitle("Strong selection with epistasis")
    fig.tight_layout()

Dominance
---------

We again assume fitness effects are the same at both loci, and now explore how dominance
affects LD. We'll start by looking at the "simple" dominance model without epistasis, so
that fitness effects are additive across loci. When simulating with dominance, the selection
model no longer collapses to a haploid model, but instead we need to specify the selection
coefficients for each possible diploid haplotype pair `AB/AB`, `AB/Ab`, etc. We'll use
another helper function to generate those selection coefficients and pass them to the
``sel_params_general`` keyword argument.

For example, to simulate the equilibrium distribution with selection coefficient -5 and
dominance coefficient 0.1 under the simple dominance model:

.. code-block:: python

    gamma = -5
    h = 0.1
    sel_params = moments.TwoLocus.Util.simple_dominance(gamma, h=h)
    F = moments.TwoLocus.Demographics.equilibrium(n, rho, sel_params_general=sel_params)

Let's look at how :math:`\sigma_d^2` and :math:`\sigma_d^1` are affected by dominance.

.. jupyter-execute::
    :hide-code:

    # precomputed with n = 50, using
    # F = moments.TwoLocus.Demographics.equilibrium(
    #     n, rho=rho, sel_params_general=moments.TwoLocus.Util.simple_dominance(gamma, h=h))
    rhos = np.logspace(-1, np.log10(30), 15)
    hs = [0, 0.1, 0.25, 0.5, 1.0]
    gammas = [-0.1, -1, -5]

    sigma_d2s = {
        -0.1: {
            0: [0.436383, 0.42619, 0.411747, 0.391813, 0.365282, 0.331625, 0.291433, 0.246751, 0.200836, 0.157269, 0.118903, 0.087235, 0.062446, 0.043837, 0.030307],
            0.1: [0.434072, 0.423973, 0.409661, 0.389902, 0.363592, 0.330199, 0.290295, 0.245899, 0.200239, 0.156878, 0.118662, 0.087094, 0.062368, 0.043795, 0.030286],
            0.25: [0.430616, 0.420658, 0.406541, 0.387042, 0.361063, 0.328062, 0.288588, 0.24462, 0.199343, 0.156291, 0.118299, 0.086883, 0.062251, 0.043733, 0.030254],
            0.5: [0.424888, 0.415162, 0.401365, 0.382295, 0.356861, 0.324509, 0.285747, 0.242489, 0.197848, 0.155309, 0.117693, 0.086529, 0.062054, 0.043628, 0.0302],
            1.0: [0.413535, 0.404256, 0.39108, 0.372838, 0.348456, 0.317356, 0.279969, 0.238085, 0.194688, 0.15317, 0.116318, 0.08568, 0.061542, 0.043317, 0.030004]
        },
        -1: {
            0: [0.433969, 0.422819, 0.40719, 0.385925, 0.358124, 0.323575, 0.283199, 0.239197, 0.194666, 0.152783, 0.115974, 0.085487, 0.061468, 0.043302, 0.030008],
            0.1: [0.413552, 0.404279, 0.391109, 0.372878, 0.348514, 0.317439, 0.280082, 0.238228, 0.194851, 0.153336, 0.116472, 0.085814, 0.061656, 0.043415, 0.03009],
            0.25: [0.375311, 0.366692, 0.354528, 0.337825, 0.31572, 0.28783, 0.25464, 0.217717, 0.179518, 0.142772, 0.10976, 0.081857, 0.059467, 0.042265, 0.029509],
            0.5: [0.321996, 0.315453, 0.306156, 0.293273, 0.276013, 0.253894, 0.22707, 0.196579, 0.164287, 0.132469, 0.103206, 0.077919, 0.057223, 0.041044, 0.02887],
            1.0: [0.233853, 0.230215, 0.224984, 0.217612, 0.207508, 0.194166, 0.177372, 0.157425, 0.13524, 0.112228, 0.089951, 0.069741, 0.052447, 0.038391, 0.027459]
        },
        -5: {
            0: [0.267953, 0.2434, 0.220828, 0.200881, 0.183373, 0.167517, 0.152268, 0.136659, 0.120087, 0.102524, 0.084556, 0.067185, 0.05146, 0.038123, 0.028654],
            0.1: [0.190778, 0.178936, 0.167025, 0.155608, 0.144876, 0.134618, 0.124339, 0.113459, 0.101534, 0.08845, 0.074543, 0.060541, 0.047349, 0.035737, 0.02669],
            0.25: [0.115954, 0.113056, 0.109616, 0.105711, 0.101407, 0.096691, 0.091431, 0.085398, 0.078356, 0.070177, 0.060969, 0.051123, 0.041263, 0.032065, 0.02464],
            0.5: [0.062152, 0.061819, 0.061343, 0.060676, 0.059755, 0.058502, 0.056823, 0.054599, 0.051697, 0.048001, 0.043457, 0.038139, 0.032289, 0.026294, 0.020583],
            1.0: [0.029166, 0.029117, 0.029044, 0.028935, 0.028773, 0.028534, 0.028183, 0.027675, 0.02695, 0.025937, 0.024562, 0.022765, 0.020526, 0.017897, 0.015018]
        }
    }
    sigma_d1s = {
        -0.1: {
            0: [-0.002806, -0.002666, -0.002488, -0.002276, -0.002046, -0.001822, -0.001616, -0.001416, -0.001184, -0.000902, -0.000598, -0.000328, -0.000132, -2.2e-05, 1e-05],
            0.1: [-0.00288, -0.002742, -0.002566, -0.002354, -0.002124, -0.001896, -0.001682, -0.00147, -0.001226, -0.000932, -0.000618, -0.00034, -0.00014, -2.8e-05, 6e-06],
            0.25: [-0.003006, -0.00287, -0.002696, -0.002484, -0.00225, -0.002014, -0.001786, -0.001556, -0.001292, -0.000978, -0.000648, -0.00036, -0.000154, -3.6e-05, 2e-06],
            0.5: [-0.003254, -0.00312, -0.002944, -0.002732, -0.00249, -0.002236, -0.001982, -0.001716, -0.001412, -0.001064, -0.000706, -0.000398, -0.000176, -5e-05, -6e-06],
            1.0: [-0.00388, -0.003744, -0.003564, -0.003338, -0.003072, -0.002774, -0.00245, -0.002098, -0.001702, -0.001266, -0.00084, -0.000482, -0.000228, -8.2e-05, -2.4e-05]
        },
        -1: {
            0: [-0.280332, -0.2626, -0.238788, -0.208344, -0.171946, -0.132068, -0.092886, -0.059014, -0.033602, -0.01713, -0.007866, -0.003282, -0.001244, -0.000412, -9.8e-05],
            0.1: [-0.255178, -0.240064, -0.219634, -0.193284, -0.1614, -0.12591, -0.090334, -0.058822, -0.03449, -0.018198, -0.008694, -0.0038, -0.001532, -0.000562, -0.000178],
            0.25: [-0.222546, -0.210674, -0.194476, -0.173314, -0.147244, -0.117536, -0.086844, -0.058634, -0.035886, -0.019882, -0.01003, -0.00466, -0.002022, -0.000828, -0.000322],
            0.5: [-0.180398, -0.172442, -0.161424, -0.146726, -0.128092, -0.106024, -0.082064, -0.05865, -0.038358, -0.02288, -0.01249, -0.006306, -0.002996, -0.001376, -0.000638],
            1.0: [-0.13055, -0.126752, -0.12136, -0.113892, -0.103926, -0.091274, -0.07624, -0.059804, -0.043556, -0.029218, -0.018018, -0.010278, -0.00552, -0.00289, -0.001564]
        },
        -5: {
            0: [-0.547684, -0.427396, -0.320646, -0.232136, -0.16268, -0.110394, -0.072208, -0.04502, -0.02627, -0.01402, -0.006712, -0.00291, -0.001256, -0.000662, -0.00087],
            0.1: [-0.333748, -0.278808, -0.225394, -0.17711, -0.13601, -0.102458, -0.07565, -0.054362, -0.037526, -0.0245, -0.014968, -0.008608, -0.004834, -0.002836, -0.001904],
            0.25: [-0.152364, -0.140342, -0.126584, -0.11184, -0.09696, -0.0826, -0.069054, -0.056314, -0.044322, -0.033224, -0.023454, -0.015566, -0.00989, -0.006282, -0.004242],
            0.5: [-0.064368, -0.063422, -0.062106, -0.060322, -0.057972, -0.054962, -0.051202, -0.04661, -0.041146, -0.034886, -0.028134, -0.021458, -0.01557, -0.011028, -0.00799],
            1.0: [-0.026374, -0.026304, -0.026202, -0.026048, -0.02582, -0.025482, -0.024986, -0.02427, -0.023254, -0.021852, -0.020002, -0.017714, -0.015138, -0.01258, -0.010414]
        }
    }

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    ax.plot(rhos_ok[:-5], ohta_kimura[:-5], "k--", lw=2, label="Ohta-Kimura")
    for ii, h in enumerate(hs):
        ax.plot(rhos, sigma_d2s[-1][h], markers[ii] + "--", label=rf"$\gamma=-1$, $h={h}$")
    ax.set_title(r"$\gamma=-1$, with varying dominance")
    ax.set_ylabel(r"$\sigma_d^2$")
    ax.set_xlabel(r"$\rho$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    ax.plot(rhos_ok[:-5], ohta_kimura[:-5], "k--", lw=2, label="Ohta-Kimura")
    for ii, h in enumerate(hs):
        ax.plot(rhos, sigma_d2s[-5][h], markers[ii] + "--", label=rf"$\gamma=-5$, $h={h}$")
    ax.set_title(r"$\gamma=-5$, with varying dominance")
    ax.set_ylabel(r"$\sigma_d^2$")
    ax.set_xlabel(r"$\rho$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()

Squared LD (:math:`\sigma_d^2`) is increased for recessive variants, while pairs of 
dominant mutations reduce it below expectations for additive variants.

.. jupyter-execute::
    :hide-code:

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    ax.plot(rhos_ok[:-5], 0 * ohta_kimura[:-5], "k--", lw=2, label="Neutrality")
    for ii, h in enumerate(hs):
        ax.plot(rhos, sigma_d1s[-1][h], markers[ii] + "--", label=rf"$\gamma=-1$, $h={h}$")
    ax.set_title(r"$\gamma=-1$, with varying dominance")
    ax.set_ylabel(r"$\sigma_d^1$")
    ax.set_xlabel(r"$\rho$")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    ax.plot(rhos_ok[:-5], 0 * ohta_kimura[:-5], "k--", lw=2, label="Neutrality")
    for ii, h in enumerate(hs):
        ax.plot(rhos, sigma_d1s[-5][h], markers[ii] + "--", label=rf"$\gamma=-5$, $h={h}$")
    ax.set_title(r"$\gamma=-5$, with varying dominance")
    ax.set_ylabel(r"$\sigma_d^1$")
    ax.set_xlabel(r"$\rho$")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()

Similarly, recessive mutations lead to larger average negative signed LD. However, this
pattern also depends on the underlying selection coefficient, with LD decay curves that
can vary qualitatively for different selection coefficients and recombination rates
between loci, even when dominance is equivalent.

.. todo:: Relate to associative overdominance work, e.g. Charlesworth, and Hill-Robertson
    interference.

Gene-based dominance
--------------------

.. todo:: All the comparisons, show LD curves and expectations for signed LD, depending
    on the selection model, maybe explore how population size changes distort these
    expectations.

Non-steady-state demography
---------------------------

.. todo:: Are any of these statistics quite sensitive to bottlenecks or expansions?

.. todo:: Discussion on what we can expect to learn from signed LD-based inferences. Are
    the various selection models and demography hopelessly confounded?

References
==========

.. [Garcia]
    Garcia, Jesse A., and Kirk E. Lohmueller. "Negative linkage disequilibrium between
    amino acid changing variants reveals interference among deleterious mutations in the
    human genome." *bioRxiv* (2020).

.. [Good]
    Good, Benjamin H. "Linkage disequilibrium between rare mutations."
    *Genetics* (2022).

.. [Hudson]
    Hudson, Richard R. "Two-locus sampling distributions and their application."
    *Genetics* 159.4 (2001): 1805-1817.

.. [Ohta]
    Ohta, Tomoko, and Motoo Kimura. "Linkage disequilibrium between two segregating
    nucleotide sites under the steady flux of mutations in a finite population."
    *Genetics* 68.4 (1971): 571.

.. [Ragsdale2022]
    Ragsdale, Aaron P. "Local fitness and epistatic effects lead to distinct patterns
    of linkage disequilibrium in protein-coding genes." *Genetics* 221.4 (2022):
    iyac097.

.. [Ragsdale_Gutenkunst]
    Ragsdale, Aaron P. and Ryan N. Gutenkunst. "Inferring demographic history using
    two-locus statistics." *Genetics* 206.2 (2017): 1037-1048.

.. [Ragsdale_Gravel]
    Ragsdale, Aaron P. and Simon Gravel. "Models of archaic admixture and recent history
    from two-locus statistics." *PLoS Genetics* 15.6 (2019): e1008204.

.. [Sandler]
    Sandler, George, Stephen I. Wright, and Aneil F. Agrawal. "Using patterns of signed
    linkage disequilibria to test for epistasis in flies and plants." *bioRxiv* (2020).

.. [Sanjak]
    Sanjak, Jaleal S., Anthony D. Long, and Kevin R. Thornton. "A model of compound
    heterozygous, loss-of-function alleles is broadly consistent with observations
    from complex-disease GWAS datasets." PLoS genetics 13.1 (2017): e1006573.

.. [Sohail]
    Sohail, Mashaal, et al. "Negative selection in humans and fruit flies involves
    synergistic epistasis." *Science* 356.6337 (2017): 539-542.
