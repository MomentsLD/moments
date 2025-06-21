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
- ``statistics``: Defaults to ``None``, which assumes that all statistics are
  present and in the conventional default order. If the data is missing some
  statistics, we must specify which statistics are present using the subset of
  statistic names given by ``moments.LD.Util.moment_names(num_pops)``.
- ``normalization``: Defaults to ``0``. The index of the population to
  normalize by, which should match the population index that we normalized by
  when parsing the data.
- ``verbose``: If an integer greater than 0, prints updates of the optimization
  procedure at intervals given by that spacing.

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

    # run optimization
    opt_params, LL = moments.LD.Inference.optimize_log_fmin(
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
regions, where ``rep_data = {0: ld_stats_0, 1: ld_stats_1, ...}``, e.g., we use

.. code-block:: python

    num_boots = 100
    norm_idx = 0
    bootstrap_sets = moments.LD.Parsing.get_bootstrap_sets(
        rep_data, num_bootstraps=num_boots, normalization=norm_idx)

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

**********
References
**********

.. [Coffman2016]
    Coffman, Alec J., et al. "Computationally efficient composite likelihood
    statistics for demographic inference."
    *Molecular biology and evolution* 33.2 (2016): 591-593.
