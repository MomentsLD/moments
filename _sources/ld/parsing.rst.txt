 .. _sec_ld_parsing:

.. jupyter-execute::
    :hide-code:

    import matplotlib, matplotlib.pylab as plt
    plt.rcParams['legend.title_fontsize'] = 'xx-small'
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    matplotlib.rc('axes', labelsize=12)
    matplotlib.rc('axes', titlesize=12)
    matplotlib.rc('legend', fontsize=10)

    # fixed random seed, so that sampled data and perturbed starting parameters
    # (and so the outputs shown) are the same in every docs build
    import numpy as np
    np.random.seed(42)

=====================
Parsing LD statistics
=====================

As described in the :ref:`multi-population LD section <sec_ld>`, we are interested
in :math:`\sigma_d^2`-type statistics, which is the ratio of expectations of
:math:`D^2` and :math:`\pi_2 = p(1-p)q(1-q)`. Again, :math:`p` and :math:`q`
are the allele frequencies at the left and right loci.

To estimate these statistics from data, we take the average of each LD statistic
over all pairs of observed (biallelic) SNPs at a given recombination distance,
and then divide by the observed :math:`\pi_2` in one of the populations (the
"normalizing" population). As described below, we also use a block bootstrapping
approach to estimate variances and covariances of observed statistics at each
recombination distance, which is used in
:ref:`inference and computing confidence intervals <sec_ld_inference>`.

***************
Binned LD decay
***************

To estimate LD decay curves from the data, we bin all pairs of observed SNPs by
recombination distance. While we can bin by physical distance (bps) separating
SNPs, genetic maps are non-uniform and physical distance does not perfectly
correlate with genetic distance at small scales. If we have a recombination map
available, it is preferable to compute and compare statistics using that map.

Recombination distances and bin edges are given in Morgans: the expected number
of crossovers between two loci per generation, which for short distances is
very close to the recombination fraction :math:`r` between them. For a sense of
scale, at a typical human recombination rate of about :math:`10^{-8}` per base
pair per generation, :math:`r = 10^{-6}` corresponds to roughly 100 bp,
:math:`10^{-4}` to roughly 10 kb, and :math:`10^{-3}` to roughly 100 kb.

Recombination distance bins are defined by bin edges, which is a list or array with
length equal to the number of desired bins plus one. Bin edges should be
monotonically increasing, and are thus adjacent without gaps between bins. Thus,
bins are defined as semi-open intervals:

.. jupyter-execute::

    import moments.LD
    import numpy as np

    bin_edges = np.array([0, 1e-6, 1e-5, 1e-4])
    print("Bins:")
    for b_l, b_r in zip(bin_edges[:-1], bin_edges[1:]):
        print(f"[{b_l}, {b_r})")

There are a few considerations to keep in mind. In practice, very short distances
can be problematic, because "non-standard" evolutionary processes can distort
allele frequency correlations for tightly linked loci. For example, our evolutionary
model does not include multi-nucleotide mutations [Harris2014]_ or gene conversion
[Ardlie2001]_, both of which operate at short distances.

Thus, when working with real data we recommend omitting bins of very short
recombination distances. In practice, we typically drop bins with length less
than :math:`r=5\times 10^{-6}`, which corresponds to roughly a few hundred bp on
average in humans.

******************
Parsing from a VCF
******************

The primary function of the ``Parsing`` module is computing LD statistics from an
input VCF. There are a number of options available, but the primary inputs are
the path to the VCF file and the bins of distances separating loci. Typically, we
work in recombination distance, in which case a recombination map is also required.
If we do not have a recombination map available, we can bin by base pair distances
instead.

The function ``moments.LD.Parsing.compute_ld_statistics`` returns a dictionary with
the bins, returned statistics, populations, and `sums` of each statistic over the
provided bins. For example:

.. code-block:: python

    # bin edges of recombination distance, in Morgans
    r_bins = np.logspace(-6, -3, 7)
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_path, r_bins=r_bins, rec_map_file=map_path)

Recombination distance bins (``r_bins``) require a recombination map. Without
one, we bin by the physical distance between SNPs instead, using ``bp_bins``,
with bin edges in base pairs:

.. code-block:: python

    # bin edges of physical distance, in base pairs (100 bp to 100 kb)
    bp_bins = np.logspace(2, 5, 7)
    ld_stats = moments.LD.Parsing.compute_ld_statistics(vcf_path, bp_bins=bp_bins)

Using a recombination map
-------------------------

The input recombination map is specified as a text file, with the first column giving
positions along the chromosome and additional column(s) defining the cumulative map(s),
typically in units of cM. The header line is "Pos Map1 Map2 (etc)", and we can use
any map in the file by specifying the ``map_name``. If no map name is given, or the
specified map name does not match a genetic map in the header, we use the map in the
first column.

Typically, maps are given in units of centi-Morgans, and the default behavior is to
assume cM units. If the map is given in units of Morgans, we need to set ``cM=False``.
Either way, the map is converted to Morgans when computing the distances between
SNPs, so ``r_bins`` are always given in Morgans (for example, ``1e-6`` Morgans,
rather than the equivalent ``1e-4`` cM).

Populations and pop-file
------------------------

We often have data from more than one population, so we need to be able to specify
which samples in the VCF correspond to which populations. This is handled by passing
a file that assigns each sample to a population. For example, the population file is
written as

.. code-block::

    sample  pop
    sid_0   pop_A
    sid_1   pop_B
    sid_2   pop_A
    sid_3   pop_A
    sid_4   pop_B
    ...

Then to include the population information in the function, we also pass a list
of the populations to compute statistics for. Samples from omitted populations
are dropped from the data.

.. code-block:: python

    pops = ["pop_A", "pop_B"]
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_path,
        r_bins=r_bins,
        rec_map_file=map_path,
        pop_file=pop_file_path,
        pops=pops
    )

Masking and using bed files
---------------------------

If there are multiple chromosomes or contigs included in the VCF, we specify
which chromosome to compute statistics for by setting the ``chromosome`` flag.
We can also subset a chromosome by including a bed file, which will filter out
all SNPs that fall outside the region intervals given in the bed file. Bed files
have the format ``{chrom}\t{left_pos}\t{right_pos}``, which defines a semi-open
interval. The path to the bed file is provided with the ``bed_file`` argument.

Computing a subset of statistics
--------------------------------

Sometimes we may wish to only compute a subset of possible LD statistics. By
default, the parsing function computes all statistics possible for the number
of populations provided. Instead, we can specify the ``stats_to_compute``, which
is a list (of length 2) of lists. The first list are the LD statistics to return,
and the second list has the heterozygosity statistics to return. Statistic names
follow the convention in ``moments.LD.Util.moment_names(num_pops)``, and should
be formatted accordingly.

.. code-block:: python

    # all statistics for two populations, as [ld_stats, h_stats]
    ld_names, h_names = moments.LD.Util.moment_names(2)

    # compute only the D^2 and pi2 statistics, and all heterozygosity statistics
    stats_to_compute = [
        [s for s in ld_names if s.startswith("DD") or s.startswith("pi2")],
        h_names,
    ]
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_path,
        r_bins=r_bins,
        rec_map_file=map_path,
        pop_file=pop_file_path,
        pops=["pop_A", "pop_B"],
        stats_to_compute=stats_to_compute,
    )

The subset must include the :ref:`normalizing statistics
<sec_ld_parsing_normalization>` of the population we will normalize by (for
population 0, ``pi2_0_0_0_0`` and ``H_0_0``), as they are needed to normalize the
data. Since the data then have only some of the statistics, we pass their names,
``ld_stats["stats"]``, as ``statistics`` to the inference and uncertainty
functions.

Phased vs unphased data
-----------------------

We can compute LD statistics from either phased or unphased data. The default
behavior is to assume that phasing is unknown, and ``use_genotypes`` is
``True`` by default. If we want to compute LD using phased data, we set
``use_genotypes=False``, and parsing uses phased haplotypes instead. In
general, phasing errors can bias LD statistics, sometimes significantly, and
using genotypes instead of haplotypes only slightly increases uncertainty in
most cases. Therefore, we usually recommend leaving ``use_genotypes=True``.

***********************************************
Computing averages and covariances over regions
***********************************************

From ``moments.LD.Parsing.compute_ld_statistics()``, we get LD statistic sums
from the regions in a VCF, perhaps constrained by a bed file. Our strategy is
to divide our data into some large number of roughly equally sized chunks, for
example 500 regions across all 22 autosomes in human data. We then compute LD
statistics independently for each region (it helps to parallelize that step,
using a compute cluster). From those outputs, we can then compute average
statistics genome-wide, as well as covariances of statistics within each bin.
Those covariances are needed to be able to compute likelihoods and run
optimization.

The outputs of ``compute_ld_statistics`` are compiled in a dictionary, where
the keys are unique region identifiers, and items the outputs of that function.
For example:

.. code-block:: python

    region_stats = {
        0: moments.LD.Parsing.compute_ld_statistics(VCF, bed_file="region_0.bed", ...),
        1: moments.LD.Parsing.compute_ld_statistics(VCF, bed_file="region_1.bed", ...),
        2: moments.LD.Parsing.compute_ld_statistics(VCF, bed_file="region_2.bed", ...),
        ...
    }

Means and variance-covariance matrices
--------------------------------------

Mean statistics and their variance-covariance matrices are computed by calling
``bootstrap_data``, passing the region statistics dictionary, and optionally
the index of the population to normalize :math:`\sigma_d^2` statistics by. By
default, the normalizing population is the first (index 0).

.. code-block:: python

    mv = moments.LD.Parsing.bootstrap_data(region_stats)

To see what this returns, we load the output of ``bootstrap_data`` for 100
regions simulated under the isolation-with-migration model of the :ref:`example
below <sec_ld_parsing_example>`, which we also use in :ref:`Inference
<sec_ld_inference>`. ``mv`` is a dictionary:

.. jupyter-execute::

    import pickle

    with open("data/means.varcovs.split_mig.100_reps.bp", "rb") as fin:
        mv = pickle.load(fin)

    print(mv.keys())
    print("populations:", mv["pops"])
    print("number of recombination bins:", len(mv["bins"]))
    print("first bin:", mv["bins"][0])

``mv["stats"]`` gives the names of the statistics as two lists: the two-locus
(LD) statistics, and the single-locus heterozygosity statistics. Here, these are
all the statistics for two populations, in the same order as
``moments.LD.Util.moment_names(2)``:

.. jupyter-execute::

    ld_stat_names, h_stat_names = mv["stats"]
    print(len(ld_stat_names), "LD statistics:", ld_stat_names)
    print(len(h_stat_names), "heterozygosity statistics:", h_stat_names)

``mv["means"]`` has one array of LD statistics for each recombination bin,
followed by a final array of heterozygosity statistics. ``mv["varcovs"]`` has
the matching variance-covariance matrices:

.. jupyter-execute::

    print("entries in means:", len(mv["means"]))  # 10 bins, plus 1 for H
    print("LD means in each bin:", mv["means"][0].shape)
    print("heterozygosity means:", mv["means"][-1].shape)
    print("LD varcov in each bin:", mv["varcovs"][0].shape)
    print("heterozygosity varcov:", mv["varcovs"][-1].shape)

To get a particular statistic, we use its index in the list of names. For
example, :math:`\sigma_d^2` in ``deme0`` (``DD_0_0``) in the first bin, and its
standard error:

.. jupyter-execute::

    i = ld_stat_names.index("DD_0_0")
    print("DD_0_0 in the first bin:", mv["means"][0][i])
    print("standard error:", mv["varcovs"][0][i, i] ** 0.5)

.. _sec_ld_parsing_normalization:

Normalizing statistics
----------------------

The means are normalized by the normalizing population (here ``deme0``, with
index 0): the LD statistics in each bin are divided by :math:`\pi_2` in that
population (``pi2_0_0_0_0``), and the heterozygosity statistics by its
heterozygosity (``H_0_0``). As a result, these two *normalizing statistics* equal
1 in every bin, with zero variance:

.. jupyter-execute::

    i_pi2 = ld_stat_names.index("pi2_0_0_0_0")
    i_H = h_stat_names.index("H_0_0")
    print("pi2_0_0_0_0 in each bin:", [m[i_pi2] for m in mv["means"][:-1]])
    print("H_0_0:", mv["means"][-1][i_H])
    print("variance of pi2_0_0_0_0 in the first bin:", mv["varcovs"][0][i_pi2, i_pi2])

Since they carry no information, the normalizing statistics are left out when
computing likelihoods. ``moments`` handles them in the same way everywhere:

- The parsing functions ``bootstrap_data``, ``get_bootstrap_sets``, and
  ``subset_data`` always **keep** the normalizing statistics in their outputs, so
  that the data always match their list of statistic names.
- The inference, uncertainty, and plotting functions **remove** the normalizing
  statistics internally, without modifying the data passed to them. So we pass
  the parsed data to them exactly as returned, e.g. ``[mv["means"],
  mv["varcovs"]]``.

:ref:`Inference <sec_ld_inference_normalization>` describes how the inference
functions match the data to the statistics of a model.

.. note::
    Before version 1.7.0, ``get_bootstrap_sets`` and ``subset_data`` removed the
    normalizing statistics from their outputs, and their ``remove_norm_stats``,
    ``remove_Dz`` (in ``get_bootstrap_sets``), and ``normalization`` (in
    ``subset_data``) options have been removed. Data saved from those versions
    need to be recomputed, or have the normalizing statistics added back in their
    positions (they equal 1, with zero variance).

Bootstrap replicates
--------------------

To compute confidence intervals with the Godambe method, we also need bootstrap
replicates of the mean statistics, which resample regions with replacement:

.. code-block:: python

    bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(region_stats)

This returns a list of bootstrap replicates (by default, as many as there are
regions). Each replicate is formatted exactly like ``mv["means"]``, with the
same statistics, including the normalizing statistics:

.. jupyter-execute::

    with open("data/bootstrap_sets.split_mig.100_reps.bp", "rb") as fin:
        bootstrap_sets = pickle.load(fin)

    print("bootstrap replicates:", len(bootstrap_sets))
    print("entries in each replicate:", len(bootstrap_sets[0]))
    print("LD means in each bin:", bootstrap_sets[0][0].shape)
    print("pi2_0_0_0_0 in the first bin:", bootstrap_sets[0][0][i_pi2])

Using a subset of the data
--------------------------

``subset_data`` takes the output of ``bootstrap_data`` and keeps a subset of the
populations or recombination bins, and can also drop the :math:`\sigma_{Dz}`
statistics. It returns the bin edges, means, varcovs, and the list of statistics
that remain. For example, to keep the bins up to :math:`r = 10^{-4}` and drop
the Dz statistics:

.. jupyter-execute::

    r_edges, means, varcovs, stats = moments.LD.Parsing.subset_data(
        mv, ["deme0", "deme1"], r_max=1e-4, remove_Dz=True
    )
    print("bin edges:", r_edges)
    print("entries in means:", len(means))
    print(len(stats[0]), "LD statistics:", stats[0])
    print("LD means in each bin:", means[0].shape)

Since these data no longer have all the statistics for two populations, we pass
the returned ``stats`` as ``statistics`` to the inference, uncertainty, and
plotting functions, along with the new bin edges. For example:

.. code-block:: python

    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [means, varcovs], [model_func], rs=r_edges, statistics=stats
    )

When keeping a subset of populations, the statistics for the other populations
are dropped, and the remaining populations are renumbered, keeping their order in
the data. (If ``pops_to`` lists them in a different order, ``subset_data`` warns
and reorders them to match the data, so that the statistic names stay in the
standard order.) The normalizing population must be kept, since the data are
still normalized by it. If it isn't, the inference and uncertainty functions raise
an error, as the remaining data's "normalizing" statistics don't equal 1. For
example, keeping only ``deme0`` leaves the single-population statistics, with
``pi2_0_0_0_0`` and ``H_0_0`` still the normalizing statistics:

.. jupyter-execute::

    r_edges, means, varcovs, stats = moments.LD.Parsing.subset_data(mv, ["deme0"])
    print(stats)
    print("LD means in each bin:", means[0].shape)
    print("pi2_0_0_0_0 in the first bin:", means[0][stats[0].index("pi2_0_0_0_0")])

.. _sec_ld_parsing_example:

*******
Example
*******

Using `msprime <https://tskit.dev/msprime/docs/latest/intro.html>`_
[Kelleher2016]_, we'll simulate some data under an isolation-with-migration
(IM) model and then compute LD and heterozygosity statistics using the
``LD.Parsing`` methods. First, the simulation will use the
``demes``-``msprime`` interface, which are then written as a VCF.

The YAML-file specifying the model is

.. literalinclude:: ../data/im-parsing-example.yaml
   :language: yaml

And we use msprime to simulate 1Mb of data, using a constant recombination
and mutation rate.

.. code-block:: python
    
    import msprime
    import demes
    import os

    # set up simulation parameters
    L = 1e6
    u = r = 1.5e-8
    n = 10
    
    g = demes.load("data/im-parsing-example.yaml")
    demog = msprime.Demography.from_demes(g)

    trees = msprime.sim_ancestry(
        {"deme0": n, "deme1": n},
        demography=demog,
        sequence_length=L,
        recombination_rate=r,
        random_seed=321,
    )

    trees = msprime.sim_mutations(trees, rate=u, random_seed=123)

    with open("data/im-parsing-example.vcf", "w+") as fout:
        trees.write_vcf(fout)

This simulation had 10 diploid individuals per population, and
``msprime``/``tskit`` writes their IDs as ``tsk_0``, ``tsk_1``, etc:

.. literalinclude:: ../data/im-parsing-example.vcf
    :lines: 1-10

To parse this data, we need the file that maps samples to populations and
the recombination map file (the total map length is found by
:math:`1 \times 10^{6} \text{ bp} \times 1.5 \times 10^{-8} \text{ M/bp} \times 100 \text{ cM/M}`):

.. literalinclude:: ../data/im-parsing-example.samples.pops.txt

.. literalinclude:: ../data/im-parsing-example.map.txt

With all this, we can now compute LD based on recombination distance bins:

.. code-block:: python

    r_bins = np.array(
        [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    )

    vcf_file = "data/im-parsing-example.vcf"
    map_file = "data/im-parsing-example.map.txt"
    pop_file = "data/im-parsing-example.samples.pops.txt"
    pops = ["deme0", "deme1"]
    ld_stats = moments.LD.Parsing.compute_ld_statistics(
        vcf_file,
        rec_map_file=map_file,
        pop_file=pop_file,
        pops=["deme0", "deme1"],
        r_bins=r_bins,
        report=False,
    )

.. jupyter-execute::
    :hide-code:

    import pickle
    r_bins = np.array([0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    with open("data/im-parsing-example.ld_stats.bp", "rb") as fin:
        ld_stats = pickle.load(fin)

The output, ``ld_stats``, is a dictionary with the keys ``bins``, ``stats``,
``pops``, and ``sums``. ``sums`` has the sum of each statistic over all pairs of
SNPs in each recombination bin, followed by the sums of heterozygosity statistics:

.. jupyter-execute::

    print(ld_stats.keys())
    print("populations:", ld_stats["pops"])
    print("number of bins:", len(ld_stats["bins"]))
    print("entries in sums:", len(ld_stats["sums"]))  # one per bin, plus 1 for H
    print("LD sums in each bin:", ld_stats["sums"][0].shape)

To get the average statistics over multiple regions
(here, we only have a single region that we simulated), we use
``means_from_region_data``:

.. jupyter-execute::

    means = moments.LD.Parsing.means_from_region_data(
        {0: ld_stats}, ld_stats["stats"], norm_idx=0
    )
    
This provides :math:`\sigma_d^2`-type statistics relative to :math:`\pi_2` in
``deme0``, and relative heterozygosities (also relative to ``deme0``).
These statistics were computed from only a single relatively small region,
so they will be quite noisy. But we can still compare to expectations under
the input IM demographic model.

.. jupyter-execute::

    import demes
    g = demes.load("data/im-parsing-example.yaml")

    y = moments.Demes.LD(
        g,
        sampled_demes=["deme0", "deme1"],
        rho=4 * g["anc"].epochs[0].start_size * r_bins,
    )
    
    # stats are computed at the bin edges - average to get midpoint estimates
    y = moments.LD.LDstats(
        [(y_l + y_r) / 2 for y_l, y_r in zip(y[:-2], y[1:-1])] + [y[-1]],
        num_pops=y.num_pops,
        pop_ids=y.pop_ids,
    )
    y = moments.LD.Inference.sigmaD2(y)

    # plot LD decay curves for some statistics
    moments.LD.Plotting.plot_ld_curves_comp(
        y,
        means[:-1],
        [],
        rs=r_bins,
        stats_to_plot=[
            ["DD_0_0", "DD_0_1", "DD_1_1"],
            ["Dz_0_0_0", "Dz_0_1_1", "Dz_1_1_1"],
            ["pi2_0_0_1_1", "pi2_0_1_0_1", "pi2_1_1_1_1"]
        ],
        labels=[[r"$D_0^2$", r"$D_0 D_1$", r"$D_1^2$"],
            [r"$Dz_{0,0,0}$", r"$Dz_{0,1,1}$", r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$", r"$\pi_{2;0,1,0,1}$", r"$\pi_{2;1,1,1,1}$"]
        ],
        plot_vcs=False,
        fig_size=(8, 3),
        show=False,
    );

Bootstrapping over multiple regions
-----------------------------------

Normally, we'll want more data than from a single 1Mb region to compute
averages and variances of statistics. Using the same approach as the above
example, ``ld_stats`` for 100 replicates we computed (see example in the
``moments`` repository `here
<https://github.com/MomentsLD/moments/tree/main/examples/LD>`_). From
this, each replicate set of statistics were placed in a dictionary, as
``rep_stats = {0: ld_stats_0, 1: ld_stats_1, ..., 99: ld_stats_99}``. This
dictionary can then be used to compute means and covariances of statistics.

.. code-block:: python

    mv = moments.LD.Parsing.bootstrap_data(rep_stats)

.. jupyter-execute::
    :hide-code:

    with open("data/means.varcovs.split_mig.100_reps.bp", "rb") as fin:
        mv = pickle.load(fin)

By simulating more data, the LD decay curves are much less noisy, and by
simulating multiple replicates, we also compute the variance-covariance
matrices for each bin and can include standard errors in the plots.

.. jupyter-execute::

    # plot LD decay curves for some statistics
    moments.LD.Plotting.plot_ld_curves_comp(
        y,
        mv["means"][:-1],
        mv["varcovs"][:-1],
        rs=r_bins,
        stats_to_plot=[
            ["DD_0_0", "DD_0_1", "DD_1_1"],
            ["Dz_0_0_0", "Dz_0_1_1", "Dz_1_1_1"],
            ["pi2_0_0_1_1", "pi2_0_1_0_1", "pi2_1_1_1_1"]
        ],
        labels=[[r"$D_0^2$", r"$D_0 D_1$", r"$D_1^2$"],
            [r"$Dz_{0,0,0}$", r"$Dz_{0,1,1}$", r"$Dz_{1,1,1}$"],
            [r"$\pi_{2;0,0,1,1}$", r"$\pi_{2;0,1,0,1}$", r"$\pi_{2;1,1,1,1}$"]
        ],
        plot_vcs=True,
        fig_size=(8, 3),
        show=False,
    );

.. note::

    The means-covariances data is required for inference using LD statistics.
    In :ref:`Inferring demography with LD <sec_ld_inference>`, we'll use the
    same ``mv`` data dictionary to refit the IM model as an example.

********************************
LD statistics in genotype blocks
********************************

``moments.LD.Parsing`` also includes some functions for computing LD from
genotype (or haplotype) blocks. Genotype blocks are arrays of shape
:math:`L\times n`, where *L* is the number of loci and *n* is the sample size.
We assume a single population, and so we compute :math:`D^2`, :math:`Dz`,
:math:`\pi_2`, and :math:`D`, either pairwise or averaged over all pairwise
comparisons.

If we have a genotype matrix containing *n* diploid samples, genotypes are
coded as 0, 1, and 2, and we set ``genotypes=True``. If we have a haplotype
matrix with data from *n* haploid copies, genotypes are coded as 0 and 1 only,
and we set ``genotypes=False``.

For example, given a single genotype matrix, we compute all pairwise statistics
and average statistics as shown below:

.. code-block:: python

    L = 10
    n = 5
    G = np.random.randint(3, size=L * n).reshape(L, n)

    # all pairwise comparisons:
    D2_pw, Dz_pw, pi2_pw, D_pw = moments.LD.Parsing.compute_pairwise_stats(G)

    # averages:
    D2_ave, Dz_ave, pi2_ave, D_ave = moments.LD.Parsing.compute_average_stats(G)

Similarly, we can compute the pairwise or average statistics between two
genotype matrices. The matrices can have differing number of loci, but they
must have the same number of samples, as the genotype matrices are assumed to
come from different regions within the same samples.

.. code-block:: python

    L2 = 12
    n = 5
    
    G2 = np.random.randint(3, size=L2 * n).reshape(L2, n)

    # all pairwise comparisons:
    D2_pw, Dz_pw, pi2_pw, D_pw = moments.LD.Parsing.compute_pairwise_stats_between(G, G2)

    # averages:
    D2_ave, Dz_ave, pi2_ave, D_ave = moments.LD.Parsing.compute_average_stats_between(G, G2)

.. note::

    Computing LD in genotype blocks uses C-extensions that are not built by
    default, so are only available if these are built when compiling the
    C-extensions. In order to use these methods, we need to build these
    extensions using the ``--ld_extensions`` flag, as ``python setup.py
    build_ext --ld_extensions -i``.

**********
References
**********

.. [Ardlie2001]
    Ardlie, Kristin, et al. "Lower-than-expected linkage disequilibrium between
    tightly linked markers in humans suggests a role for gene conversion."
    *The American Journal of Human Genetics* 69.3 (2001): 582-589.

.. [Harris2014]
    Harris, Kelley, and Rasmus Nielsen. "Error-prone polymerase activity causes
    multinucleotide mutations in humans." *Genome research* 24.9 (2014): 1445-1454.

.. [Kelleher2016]
    Kelleher, Jerome, Alison M. Etheridge, and Gilean McVean. "Efficient
    coalescent simulation and genealogical analysis for large sample sizes."
    *PLoS computational biology* 12.5 (2016): e1004842.
