 .. _sec_ld_inference:

============================
Inferring demography with LD
============================

As described in the :ref:`linkage disequilibrium <sec_ld>` and :ref:`LD Parsing
<sec_ld_parsing>` sections, we use a family of normalized LD and heterozygosity
statistics to compare between model expectations and data. We optimize
demographic model parameters to find the expected binned LD and heterozygosity
statistics that maximize a composite likelihood over all pairs of SNPs and
recombination bins.

In this section, we'll describe the likelihood framework, how to define
demographic models that can be used in inference, how to run optimization using
``moments``' built-in inference functions, and how to compute confidence
intervals. We include a short example, following the :ref:`parsing
<sec_ld_parsing_example>` of data simulated under an isolation-with-migration
model, to illustrate the main features and options.

********************
Likelihood framework
********************

For a given recombination distance bin indexed by :math:`i`, we have a set of
computed LD statistic means :math:`D_i` from data along with the
variance-covariance matrix :math:`\Sigma_i` as returned by
``moments.LD.Parsing.bootstrap_data``. We assume a multivariate Guassian
likelihood function, so that a model parameterized by :math:`\Theta` that has
expected statistics :math:`\mu_i(\Theta)` has likelihood

.. math::
    \mathcal{L}_i(\Theta | D_i) = P(D_i | \mu_i, \Sigma_i)
    = \frac{\exp\left(-\frac{1}{2}(D_i-\mu_i)^T\Sigma_i^{-1}(D_i-\mu_i)\right)}{(2\pi)^{k/2}|\Sigma_i|^{1/2}}.

The likelihood is computed similarly for heterozygosity statistics, given their
variance-covariance matrix. Then the composite likelihood of two-locus data
across recombination bins and single-locus heterozygosity (indexed by
:math:`i=n+1` where :math:`n` is the total number of recombination bins), is

.. math::
    \mathcal{L} = \prod_{i=1}^{n+1}\mathcal{L}_i.

In practice, we work with the log of the likelihood, so that products turn to
sums and we can drop constant factors:

.. math::
    \log\mathcal{L} \propto -\frac{1}{2}\sum_{i=1}^{n+1} (D_i-\mu_i)^T\Sigma_i^{-1}(D_i-\mu_i).

As the data :math:`\{D_i, \Sigma_i\}` is fixed, we search for the model
parameters :math:`\Theta` that provide :math:`\{\mu_i\}` that maximizes
:math:`\log\mathcal{L}`.

***************************
Defining demographic models
***************************

There are a handful of built-in demographic models for one-, two-, and
three-population scenarios that can be used in inference (see :ref:`here
<sec_api_ld_demog>`). However, these are far from comprehensive and it is
likely that custom demographic models will need to be written for a given
inference problem. For inspiration, ``moments.LD.Demographics1D``,
``Demographics2D``, and ``Demographics3D`` can be used as starting points and
as illustrations of how to structure model functions.

Demographic models all require a ``params`` positional argument and ``rho`` and
(optionally) ``theta`` keyword arguments. ``theta``, the population-size scaled
mutation rate, does not play a role in inference using relative statistics, as
the mutation rate cancels in :math:`\sigma_d^2`-type statistics. 

For example, the IM model we simulated data under in the LD Parsing section
could be parameterized as

.. jupyter-execute::

    def model_func(params, rho=None, theta=0.001):
        nu0, nu1, T, M = params
        y = moments.LD.Numerics.steady_state([1], rho=rho, theta=theta)
        y = moments.LD.LDstats(y, num_pops=1)
        y = y.split(0)
        y.integrate([nu0, nu1], T, m=[[0, M], [M, 0]], rho=rho, theta=theta)
        return y

In the input demographic model to the simulations, we had the ancestral
effective population size as 10,000, the size of deme0 was 2,000, and the size
of deme1 was 20,000. The populations split 1,500 generations ago, and exchanged
migrants symmetrically at a rate of 0.0001 per-generation. Converted into
genetic units, ``nu0 = 0.2``, ``nu1 = 2``, ``T=1500 / 2 / 10000 = 0.075``, and
``M = 2 * 10000 * 0.0001 = 2.0``.

********************
Running optimization
********************

Optimization with ``moments.LD``, much like ``moments`` optimization with the
SFS, includes a handful functions that serve as wrappers for ``scipy``
optimizers with options specific to working with LD statistics. The two primary
functions in ``moments.LD.Inference`` are

- ``optimize_log_fmin``: Uses the downhill simplex algorithm on the log of
  the parameters.
- ``optimize_log_powell``: Uses the modified Powell’s method, which optimizes
  slices of parameter space sequentially.

Each optimization method accepts the same arguments. Required positional
arguments are

- ``p0``: The initial guess for the parameters in ``model_func``.
- ``data``: Structured as a list of lists of data means and data var-cov
  matrices. I.e., ``data = [[means[0], means[1], ...], [varcovs[0], varcovs[1],
  ...]]``, with the final entry of the lists the means and varcovs of the
  heterozygosity statistics.
- ``model_func``: The demographic model to be fit (see above section).
  Importantly, this is a *list*, where the first entry is the LD model, which
  is always used, and the optional second entry is a demographic model for the
  SFS (which is a rarely used option and can be ignored). So usually, we would
  set ``model_func`` as ``[model_func_ld]``.

Additionally, we will almost always pass the list of unscaled recombination
bin edges as ``rs = [r0, r1, ..., rn]``, which defines *n* recombination bins.

The effective population size plays a different role in LD inference than it
does in SFS inference. For the site frequency spectrum, :math:`N_e` merely acts
as a linear scaling factor and is absorbed by the scaled mutation rate
:math:`\theta`, which is treated as a free parameter. Here, :math:`N_e` instead
rescales recombination rates, and because we use a recombination map to
determine the binning of data by recombination distances separating loci,
:math:`N_e` is a parameter that must be either passed as a fixed value or
simultaneously fit in the optimization.

If ``Ne`` is a fixed value, we specify the population size using that keyword
argument. Otherwise, if ``Ne`` is to be fit, our list of parameters to fit by
convention includes ``Ne`` in the final position in the list. Typically, ``Ne``
is not a parameter of the demographic model, as we work in rescaled genetic
units, so the parameters that get passed to ``model_func`` are ``params[:-1]``.
However, it is also possible to write a demographic model that also uses ``Ne``
as a parameter. In this case we set ``pass_Ne`` to ``True``, so that ``Ne``
both rescales recombination rates and is a model parameter, and all ``params``
are passed to ``model_func``.

- ``Ne``: The effective population size, used to rescale ``rs`` to get ``rhos
  = 4 * Ne * rs``.
- ``pass_Ne``: Defaults to ``False``. If ``True``, the demographic model
  includes ``Ne`` as a parameter (in the final position of input parameters).

Other commonly used options include

- ``fixed_params``: Defaults to ``None``. To fix some parameters, this should
  be a list of equal length as ``p0``, with ``None`` for parameters to be fit
  and fixed values at corresponding indexes.
- ``lower_bound``: Defaults to ``None``. Constraints on the lower bounds during
  optimization. These are given as lists of the same length of the parameters.
- ``upper_bound``: Defaults to ``None``. Constraints on the upper bounds during
  optimization. These are given as lists of the same length of the parameters.
- ``statistics``: Defaults to ``None``, which assumes that the data contain the
  full set of statistics for the number of populations in the model, as returned
  by the parsing functions. If the data contain a subset of statistics, we list
  them here. See :ref:`below <sec_ld_inference_normalization>`.
- ``normalization``: Defaults to ``0``. The index of the population to
  normalize by, which should match the population index that we normalized by
  when parsing the data.
- ``verbose``: If an integer greater than 0, prints updates of the optimization
  procedure at intervals given by that spacing.

.. note::
    ``verbose=1`` prints a line for every function evaluation, which can be many
    lines per second. Displaying them in a visible terminal window can slow down
    the optimization, because the program waits for the terminal to draw each
    line (it may run much faster when the window is minimized). Use a larger
    value, such as 10 to 100, to avoid this.

.. _sec_ld_inference_normalization:

Normalizing statistics and the ``statistics`` option
----------------------------------------------------

The data are normalized by :math:`\pi_2` and :math:`H` in the normalizing
population ``k`` (see :ref:`Parsing <sec_ld_parsing_normalization>`), so the
normalizing statistics ``pi2_k_k_k_k`` and ``H_k_k`` equal 1 in both the data and
the model, and have zero variance. They carry no information, so they are left out
when computing likelihoods. This is handled the same way by all LD inference,
uncertainty, and plotting functions, in both ``moments.LD`` and ``moments.Demes``:

- The data passed to these functions should include the normalizing statistics,
  as returned by the parsing functions. They are removed internally, and the data
  passed in are not modified.
- If ``statistics`` is not given (the default), the data must contain the full set
  of statistics for the number of populations, in the order given by
  ``moments.LD.Util.moment_names(num_pops)``. Data with any other number of
  statistics raise an error.
- If the data contain a subset of statistics, ``statistics`` must list exactly
  those statistics, as ``[ld_stats, h_stats]``, in the same order as the data.
  This must be the standard order of ``moments.LD.Util.moment_names(num_pops)``,
  which is the order returned by the parsing functions. For example,
  ``moments.LD.Parsing.subset_data`` returns the list of statistics along with the
  subset data. If the normalizing statistics are in the list, they are removed
  along with the data.
- The normalizing statistics in the data must equal 1. If they don't, the data
  were normalized by a different population than ``normalization`` (for example,
  if the normalizing population was left out with ``subset_data``), and an error
  is raised.
- In ``moments.LD`` functions, ``normalization`` is the *index* of the normalizing
  population. In ``moments.Demes`` functions, it is the population's *name*.

For example, with the data ``mv`` returned by ``bootstrap_data``:

.. code-block:: python

    # all statistics, as returned by bootstrap_data, so statistics isn't needed
    opt_params, neg_LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [mv["means"], mv["varcovs"]], [model_func], rs=r_bins
    )

    # a subset of the statistics, without Dz, so we pass their names
    r_edges, means, varcovs, stats = moments.LD.Parsing.subset_data(
        mv, ["deme0", "deme1"], remove_Dz=True
    )
    opt_params, neg_LL = moments.LD.Inference.optimize_log_fmin(
        p_guess, [means, varcovs], [model_func], rs=r_edges, statistics=stats
    )

.. note::
    Before version 1.7.0, ``get_bootstrap_sets`` and ``subset_data`` removed the
    normalizing statistics from their outputs. Data saved from those versions need
    to be recomputed, or have the normalizing statistics added back in their
    positions (they equal 1, with zero variance).

Example
-------

Using the data simulated in the :ref:`Parsing <sec_ld_parsing>` section, we can
refit the demographic model under a parameterized IM model. For this, we could
use the ``moments.LD.Demographics2D.split_mig`` model as our ``model_func``,
which is equivalent to the function we defined above (which we use in this
example). After loading the data and setting up the inference options, we'll
use ``optimize_log_fmin`` to fit the model.

.. jupyter-execute::

    import moments.LD
    import pickle

    with open("data/means.varcovs.split_mig.100_reps.bp", "rb") as fin:
        data = pickle.load(fin)

    rs = [0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]

    p_guess = [0.1, 2.0, 0.075, 2.0, 10000]
    p0 = moments.LD.Util.perturb_params(p_guess, fold=0.2)

    # run optimization, which returns the negative log-likelihood
    opt_params, neg_LL = moments.LD.Inference.optimize_log_fmin(
        p_guess,
        [data["means"], data["varcovs"]],
        [model_func],
        rs=rs,
        verbose=40,
    )

    # get physical units, rescaling by Ne
    physical_units = moments.LD.Util.rescale_params(
        opt_params, ["nu", "nu", "T", "m", "Ne"]
    )
    
    print("best fit parameters:")
    print(f"  N(deme0)         :  {physical_units[0]:.1f}")
    print(f"  N(deme1)         :  {physical_units[1]:.1f}")
    print(f"  Div. time (gen)  :  {physical_units[2]:.1f}")
    print(f"  Migration rate   :  {physical_units[3]:.6f}")
    print(f"  N(ancestral)     :  {physical_units[4]:.1f}")

These should be pretty close to the input demographic parameters from the
simulations. They won't be spot on, as this was only using 100Mb of simulated
data, but we should be in the ballpark.

******************************
Computing confidence intervals
******************************

When running demographic inference, we get a point estimate for the *best fit*
demographic parameters. However, for an unknown underlying true value, it's
important to also estimate what's called a confidence interval. The CI tells us
the probability that the true value lies within some range, and provides some
information about which parameters in our demographic model are tightly
constrained and which parameters we have little power to pin down.

``moments.LD`` can estimate confidence intervals using either the Fisher
Information Matrix (FIM) or the Godambe Information Matrix (GIM). In almost all
cases when using real data (or even most simulated data), the FIM will estimate
a much smaller CI than the GIM. This occurs because the FIM assumes all data
points that we've used are independent, when in reality there is linkage that
causes data points to be sometimes highly correlated between pairs of loci and
between recombination bins. The Godambe method uses bootstrap-resampled
replicates of the data to account for this correlation and does a much better
job at estimating the true underlying CIs [Coffman2016]_.

.. note::

    If you use the Godambe approach to estimate confidence intervals, please
    cite [Coffman2016]_. Alec originally implemented this approach in ``dadi``,
    and ``moments`` has more-or-less used this same implementation here.

To create bootstrap replicates from the dictionary of data sums computed over
regions, where ``region_stats = {0: ld_stats_0, 1: ld_stats_1, ...}``, e.g., we use

.. code-block:: python

    num_boots = 100
    norm_idx = 0
    bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(
        region_stats, num_bootstraps=num_boots, normalization=norm_idx)

The bootstrap sets have the same statistics as the means and varcovs from
``bootstrap_data``, including the normalizing statistics, which are removed when
computing uncertainties (see :ref:`above <sec_ld_inference_normalization>`).

These bootstrap sets can then be used as the inputs to the ``moments.LD.Godambe``
methods. The two CI estimation methods are

- ``FIM_uncert``: Uses the Fisher Information Matrix. Usage is
  ``FIM_uncert(model_func, opt_params, means, varcovs, r_edges=rs)``. 
- ``GIM_uncert``: Uses the Godambe Information Matrix. Usage is ``GIM_uncert(model_func, bootstrap_sets, opt_params, means, varcovs, r_edges=rs)``.

In each case, the model function is the same as used in inference (some
manipulation may be needed if we had any fixed parameters), means and varcovs
are the same data as input to the inference function, and ``r_edges`` are the
bin edges used in the inference. Additional options for some corner cases are
described in the :ref:`API reference for LD methods <sec_api_ld>`.

Example
-------

We'll use both the FIM and GIM to compute uncertainties from the above example
inference.

Using the FIM approach:

.. jupyter-execute::

    # using FIM
    uncerts_FIM = moments.LD.Godambe.FIM_uncert(
        model_func,
        opt_params,
        data["means"],
        data["varcovs"],
        r_edges=rs,
    )

    # lower and upper CIs, in genetic units
    lower = opt_params - 1.96 * uncerts_FIM
    upper = opt_params + 1.96 * uncerts_FIM

    # convert to physical units
    lower_pu = moments.LD.Util.rescale_params(lower, ["nu", "nu", "T", "m", "Ne"])
    upper_pu = moments.LD.Util.rescale_params(upper, ["nu", "nu", "T", "m", "Ne"])

    print("95% CIs:")
    print(f"  N(deme0)         :  {lower_pu[0]:.1f} - {upper_pu[0]:.1f}")
    print(f"  N(deme1)         :  {lower_pu[1]:.1f} - {upper_pu[1]:.1f}")
    print(f"  Div. time (gen)  :  {lower_pu[2]:.1f} - {upper_pu[2]:.1f}")
    print(f"  Migration rate   :  {lower_pu[3]:.6f} - {upper_pu[3]:.6f}")
    print(f"  N(ancestral)     :  {lower_pu[4]:.1f} - {upper_pu[4]:.1f}")

And using the GIM approach:

.. jupyter-execute::

    with open("data/bootstrap_sets.split_mig.100_reps.bp", "rb") as fin:
        bootstrap_sets = pickle.load(fin)

    # using GIM
    uncerts_GIM = moments.LD.Godambe.GIM_uncert(
        model_func,
        bootstrap_sets,
        opt_params,
        data["means"],
        data["varcovs"],
        r_edges=rs,
    )

    # lower and upper CIs, in genetic units
    lower = opt_params - 1.96 * uncerts_GIM
    upper = opt_params + 1.96 * uncerts_GIM

    # convert to physical units
    lower_pu = moments.LD.Util.rescale_params(lower, ["nu", "nu", "T", "m", "Ne"])
    upper_pu = moments.LD.Util.rescale_params(upper, ["nu", "nu", "T", "m", "Ne"])

    print("95% CIs:")
    print(f"  N(deme0)         :  {lower_pu[0]:.1f} - {upper_pu[0]:.1f}")
    print(f"  N(deme1)         :  {lower_pu[1]:.1f} - {upper_pu[1]:.1f}")
    print(f"  Div. time (gen)  :  {lower_pu[2]:.1f} - {upper_pu[2]:.1f}")
    print(f"  Migration rate   :  {lower_pu[3]:.6f} - {upper_pu[3]:.6f}")
    print(f"  N(ancestral)     :  {lower_pu[4]:.1f} - {upper_pu[4]:.1f}")

We can see above that the FIM uncertainties are considerably smaller (i.e. more
constrained) than the GIM uncertainties. However, the GIM uncertainties are to
be preferred here, as they more accurately estimate the underlying true
uncertainty in the demographic inference.

.. _sec_ld_inference_lrt:

***********************
Likelihood ratio tests
***********************

To test whether a more complex model fits the data significantly better than a
simpler, nested model (one that is a special case of the complex model with some
parameters fixed), we can use a likelihood ratio test. The test statistic is
twice the difference in log-likelihoods of the best fits,
:math:`D = 2 (\ell_{complex} - \ell_{simple})`. If the data points were
independent, :math:`D` would follow a :math:`\chi^2` distribution under the
simple model, with degrees of freedom equal to the number of nested parameters.

The LD likelihood is a composite likelihood, which treats the recombination
bins as independent, so this :math:`\chi^2` test is badly overconfident.
Following [Coffman2016]_, ``moments.LD.Godambe.LRT_adjust`` uses the bootstrap
replicates to compute an adjustment factor ``adj``, and the adjusted statistic
:math:`D_{adj} = adj \times D` is then compared to the :math:`\chi^2`
distribution. For LD statistics, ``adj`` is usually much smaller than one,
because the statistics in different recombination bins are computed from the
same pairs of SNPs and so are strongly correlated.

``LRT_adjust`` takes the complex model function and the parameters of the best
fit of the simple model, given as parameters of the complex model with the nested
parameters set to their fixed values (and Ne as the last entry), along with the
indices of the nested parameters:
``LRT_adjust(model_func, bootstrap_sets, p_nested, nested_indices, means,
varcovs, r_edges=rs)``. The p-value is given by ``moments.Godambe.sum_chi2_ppf``.

Example
-------

We'll test whether migration is needed to explain the data from the example
above, by fitting a model without migration, in which the migration rate is
fixed at zero. Because zero is on the boundary of the allowed range of the
migration rate, the test statistic follows an equal mixture of :math:`\chi^2`
distributions with zero and one degrees of freedom, which we specify with
``weights=(0.5, 0.5)``.

.. jupyter-execute::

    def model_nomig(params, rho=None, theta=0.001):
        nu0, nu1, T = params
        return model_func([nu0, nu1, T, 0], rho=rho, theta=theta)

    opt_params_nomig, neg_LL_nomig = moments.LD.Inference.optimize_log_fmin(
        [0.1, 2.0, 0.075, 10000],
        [data["means"], data["varcovs"]],
        [model_nomig],
        rs=rs,
    )

    # the best fit without migration, as parameters of the model with migration
    p_nested = [*opt_params_nomig[:3], 0, opt_params_nomig[3]]
    adj = moments.LD.Godambe.LRT_adjust(
        model_func,
        bootstrap_sets,
        p_nested,
        [3],
        data["means"],
        data["varcovs"],
        r_edges=rs,
    )

    D = 2 * (neg_LL_nomig - neg_LL)
    D_adj = adj * D
    print(f"Test statistic D  : {D:.2f}")
    print(f"Adjustment factor : {adj:.4f}")
    print(f"Adjusted D        : {D_adj:.2f}")
    p_unadj = moments.Godambe.sum_chi2_ppf(D, weights=(0.5, 0.5))
    p_adj = moments.Godambe.sum_chi2_ppf(D_adj, weights=(0.5, 0.5))
    print(f"p-value, unadjusted: {p_unadj:.3g}")
    print(f"p-value, adjusted  : {p_adj:.3g}")

Without the adjustment, the p-value is effectively zero. After adjusting for
the correlations between recombination bins, the test still rejects the model
without migration, which we know to be the right answer for these simulated data,
but with a much less extreme p-value.

.. _sec_lrt_conservative:

When the adjustment is too conservative
---------------------------------------

``LRT_adjust`` (for both the SFS and LD) can be very conservative. Its
adjustment factor shrinks roughly as :math:`1/D`, so the adjusted statistic
stays small even when :math:`D` is very large. In simulations testing for weak
migration, the adjusted SFS test rarely rejected the model without migration at
a 1% level, and the adjusted LD test did so in fewer than half of the
replicates.

A large :math:`D` (hundreds or more) with an adjusted statistic of only a few
units suggests the adjustment is too conservative, and a non-significant result
should then be treated as inconclusive. ``LRT_adjust`` also warns when the
Hessian of the complex model at the simple model's best fit is indefinite or
nearly singular, in which case the adjustment factor is unreliable. In either
case, the alternatives are, from cheapest to most expensive:

1. **A Wald test from the GIM uncertainties of the complex model.** This needs no
   further fits: compute :math:`SE_{GIM}` at the complex model's best fit with
   ``moments.LD.Godambe.GIM_uncert`` (or ``moments.Godambe.GIM_uncert`` for the
   SFS). For a parameter fixed at a boundary, such as a migration rate of zero,
   use the one-sided :math:`z = (\hat{\theta} - \theta_0) / SE_{GIM}` with
   p-value ``scipy.stats.norm.sf(z)``. Caveats: the GIM uncertainty is
   least reliable near the boundary, the false-positive rate of this test has
   not been checked, and the result depends on the parameterization (e.g., a
   rate or its log).

2. **Refitting the complex model to the bootstrap replicates.** Reject the
   simple model if the bootstrap interval of the nested parameter excludes its
   fixed value. This needs one converged optimization per replicate, which is
   the cost the Godambe methods avoid. It is an interval test, not a null
   distribution for :math:`D`, and bootstrap intervals are imperfect near a
   boundary.

3. **A parametric bootstrap under the simple model.** Simulate genomes (e.g.,
   with msprime) under the simple model's best fit, matching the data's sample
   sizes, sequence lengths, and recombination and mutation rates, and fit both
   models to each replicate. The p-value is the fraction of simulated :math:`D`
   at least as large as the observed one. This is the most reliable option but
   the most expensive. Sampling from the composite likelihood instead (Poisson
   SFS, or independent Gaussian bins for LD) ignores linkage and is far too
   liberal.

For the data above, the Wald test gives much stronger evidence for migration
than the adjusted likelihood ratio test:

.. jupyter-execute::

    import scipy.stats

    z = opt_params[3] / uncerts_GIM[3]
    print(f"Wald z            : {z:.2f}")
    print(f"p-value, one-sided: {scipy.stats.norm.sf(z):.3g}")
