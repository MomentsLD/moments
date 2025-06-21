# This script contains functions to compute the sample SFS from a demography
# defined using demes. Not ever deme graph will be supported, as moments can
# only handle integrating up to five populations, and cannot include selfing or
# cloning.

from collections import defaultdict
import math
import copy
import numpy as np
import demes
import warnings

import moments
import moments.LD


def SFS(
    g,
    sampled_demes=None,
    sample_sizes=None,
    sample_times=None,
    samples=None,
    unsampled_n=4,
    gamma=None,
    s=None,
    h=None,
    theta=None,
    u=None,
    reversible=False,
    L=1,
):
    """
    Compute the SFS from a ``demes``-specified demographic model.
    ``demes`` is a package for specifying demographic models in a
    user-friendly, human-readable YAML format. This function
    automatically parses the demographic model and returns the SFS
    for the specified populations, sample sizes, and (optionally)
    sampling times.

    Selection and dominance can be specified as a single value for
    all populations, or on a per-deme basis using a dictionary
    mapping deme name to the coefficient (defaults can also be set
    if multiple demes have the same selection or dominance
    coefficient). The mutation rate can be given as either a
    per-base rate (possibly multiplied by the sequence length),
    or as a population size-scaled rate. If mutation rates are not
    given, the SFS is scaled by ``4*N_e``, so that multiplying the
    output SFS by ``u`` results in a properly scaled SFS.

    :param g: A ``demes`` DemeGraph from which to compute the SFS.
    :type g: :class:`demes.DemeGraph`
    :param sampled_demes: A list of deme IDs to take samples from. We can repeat
        demes, as long as the sampling of repeated deme IDs occurs at distinct
        times.
    :type sampled_demes: list of strings
    :param sample_sizes: A list of the same length as ``sampled_demes``,
        giving the sample sizes for each sampled deme.
    :type sample_sizes: list of ints
    :param sample_times: If None, assumes all sampling occurs at the end of the
        existence of the sampled deme. If there are
        ancient samples, ``sample_times`` must be a list of same length as
        ``sampled_demes``, giving the sampling times for each sampled
        deme. Sampling times are given in time units of the original deme graph,
        so might not necessarily be generations (e.g. if ``g.time_units`` is years)
    :type sample_times: list of scalars, optional
    :param unsampled_n: The default sample size of unsampled demes, which must be
        greater than or equal to 4.
    :type unsampled_n: int, optional
    :param gamma: The scaled selection coefficient(s), ``2*Ne*s``. Defaults to None,
        which implies neutrality. Can be given as a scalar value, in which case
        all populations have the same selection coefficient. Alternatively, can
        be given as a dictionary, with keys given as population names in the
        input Demes model. Any population missing from this dictionary will be
        assigned a selection coefficient of zero. A non-zero default selection
        coefficient can be provided, using the key ``_default``. See the Demes
        exension documentation for more details and examples.
    :type gamma: scalar or dict
    :param h: The dominance coefficient(s). Defaults to additivity (or genic
        selection). Can be given as a scalar value, in which case all populations
        have the same dominance coefficient. Alternatively, can be given as a
        dictionary, with keys given as population names in the input Demes model.
        Any population missing from this dictionary will be assigned a dominance
        coefficient of ``1/2`` (additivity). A different default dominance
        coefficient can be provided, using the key ``_default``. See the Demes
        exension documentation for more details and examples.
    :type h: scalar or dict
    :param theta: The scaled mutation rate(s), 4*Ne*u. When simulating under the
        infinite sites model (the default mutation model), ``theta`` should be given
        as a scalar value greater than zero. If it is not provided, it is computed
        using the input value of ``u`` as ``4*Ne*u``. If ``u`` is not provided, then
        the SFS is scaled by ``4*Ne``, and the user can recover a properly scaled SFS
        by multiplying it by ``u`` or ``u*L``. When simulating under the reversible
        mutation model (with ``reversible=True``), ``theta`` may be a list of length
        2 and both the forward and backward scaled mutation rates must be less
        than 1.
    :type theta: scalar or list of length 2
    :param u: The per-base mutation rate. When simulating under the infinite sites
        model (the default mutation model), ``u`` should be a scalar. When simulating
        under the reversible mutation model (with ``reversible=True``), ``u`` may
        be a list of length 2, and mutation rate(s) must be small enough so that
        the product of ``4*Ne*u`` is less than 1.
    :type u: scalar or list of length 2
    :param L: The effective sequence length, which may be used along with ``u`` to
        set the total mutation rate. Defaults to 1, and it must be 1 when using
        the reversible mutation model.
    :type L: scalar
    :return: A ``moments`` site frequency spectrum, with dimension equal to the
        length of ``sampled_demes``, and shape equal to ``sample_sizes`` plus one
        in each dimension, indexing the allele frequency in each deme from 0
        to n[i], where i is the deme index.
    :rtype: :class:`moments.Spectrum`
    """
    # could specify samples as a dict instead of sampled_demes and sample_sizes
    if samples is None:
        if sampled_demes is None or sample_sizes is None:
            raise ValueError(
                "must specify either samples (as a dict mapping demes to sample sizes,"
                " or specify both sampled_demes and sample_times"
            )
    else:
        if type(samples) is not dict:
            raise ValueError("samples must be a dict mapping demes to sample sizes")
        if sampled_demes is not None or sample_sizes is not None:
            raise ValueError(
                "if samples is given as dict, cannot "
                "specify sampled_demes or sample_sizes"
            )
        if sample_times is not None:
            raise ValueError("if samples is given as dict, cannot specify sample times")
        sampled_demes = list(samples.keys())
        sample_sizes = list(samples.values())

    if len(sampled_demes) != len(sample_sizes):
        raise ValueError("sampled_demes and sample_sizes must be same length")
    if sample_times is not None and len(sampled_demes) != len(sample_times):
        raise ValueError("sample_times must have same length as sampled_demes")
    for deme in sampled_demes:
        if deme not in g:
            raise ValueError(f"deme {deme} is not in demography")

    # we need to copy these to new variable names
    # so they don't get updated during optimization
    sampled_pops = copy.copy(sampled_demes)
    deme_sample_times = copy.copy(sample_times)

    if unsampled_n < 4:
        raise ValueError("unsampled_n must be greater than 3")

    sampled_deme_end_times = [g[d].end_time for d in sampled_pops]
    if deme_sample_times is None:
        deme_sample_times = sampled_deme_end_times

    for t, d in zip(deme_sample_times, sampled_pops):
        if t < g[d].end_time or t >= g[d].start_time:
            raise ValueError(f"sample time {t} is outside of deme {d}'s time span")

    # for any ancient samples, we need to add frozen branches
    # with this, all "sample times" are at time 0, and ancient sampled demes are frozen
    if np.any(np.array(deme_sample_times) != 0):
        g, sampled_pops, list_of_frozen_demes = _augment_with_ancient_samples(
            g, sampled_pops, deme_sample_times
        )
        deme_sample_times = [0 for _ in deme_sample_times]
    else:
        list_of_frozen_demes = []

    if g.time_units != "generations":
        g, deme_sample_times = _convert_to_generations(g, deme_sample_times)

    # if any sample sizes are less than unsample_n, we increase and project after
    sim_sample_sizes = []
    for d, n, t in zip(sampled_pops, sample_sizes, deme_sample_times):
        sim_sample_sizes.append(max(n, unsampled_n))
        if t < g[d].end_time or t >= g[d].start_time:
            raise ValueError("sample time for {deme} must be within its time span")

    # get reference Ne from demes model
    Ne = _get_root_Ne(g)

    # if (unscaled) s is provided, convert into (scaled) gamma selection coefficients
    if s is not None:
        if gamma is not None:
            raise ValueError("Cannot specify both gamma and s")
        if isinstance(s, (int, float)):
            gamma = 2 * Ne * s
        elif type(s) is dict:
            gamma = {}
            for k, v in s.items():
                gamma[k] = 2 * Ne * v
        else:
            raise ValueError("Selection coefficient must be a scalar value or dict")

    # check selection and dominance inputs
    if gamma is not None:
        if "_default" in g:
            raise ValueError(
                "Cannot use `_default` as a deme name when selection is specified"
            )
        if isinstance(gamma, (int, float)):
            if not np.isfinite(gamma):
                raise ValueError("Selection coefficient must be a finite number")
        elif type(gamma) is dict:
            for k in gamma.keys():
                if k != "_default" and k not in g:
                    raise ValueError(f"Deme {k} in gamma, but {k} not in input graph")
        else:
            raise ValueError("Selection coefficient must be a scalar value or dict")
    if h is not None:
        if type(h) is dict:
            for k in h.keys():
                if k != "_default" and k not in g:
                    raise ValueError(f"Deme {k} in h, but {k} not in input graph")

    # set up the mutation rates as needed
    if theta is None:
        if u is None:
            u = 1
        if isinstance(u, (int, float)):
            theta = 4 * Ne * u * L
        else:
            if np.ndim(u) != 1 or len(u) != 2:
                raise ValueError(
                    "Mutation rates must be a list of length 2 when using "
                    "the reversible mutation model"
                )
            theta = [4 * Ne * u[0], 4 * Ne * u[1]]
    else:
        if u is not None:
            raise ValueError("Only one of u or theta may be specified")
        if isinstance(theta, (int, float)):
            theta *= L
        else:
            if np.ndim(theta) != 1 or len(theta) != 2:
                raise ValueError(
                    "Mutation rates must be a list of length 2 when using "
                    "the reversible mutation model"
                )
            theta[0] *= L
            theta[1] *= L

    # if a scalar, must be positive; if list-like, must be length 2 and both positive
    if not reversible:
        if not isinstance(theta, (int, float)):
            raise ValueError(
                "Mutation rate must be a scalar value for the default ISM model"
            )
        if theta <= 0:
            raise ValueError("Mutation rate must be positive")
    if reversible:
        if L != 1:
            raise ValueError(
                "Sequence length L must be 1 when using the reversible mutation model"
            )
        if isinstance(theta, (int, float)):
            theta = [theta, theta]
        if theta[0] <= 0 or theta[1] <= 0:
            raise ValueError("Mutation rates must be positive")
        if theta[0] >= 1 or theta[1] >= 1:
            raise ValueError("Mutation rates too large for reversible mutation model")

    # get the list of demographic events from demes, which is a dictionary with
    # lists of splits, admixtures, mergers, branches, and pulses
    demes_demo_events = g.discrete_demographic_events()

    # get the dict of events and event times that partition integration epochs, in
    # descending order. events include demographic events, such as splits and
    # mergers and admixtures, as well as changes in population sizes or migration
    # rates that require instantaneous changes in the size function or migration matrix.
    # get the list of demes present in each epoch, as a dictionary with non-overlapping
    # adjoint epoch time intervals
    demo_events, demes_present = _get_demographic_events(
        g, demes_demo_events, sampled_pops
    )

    for epoch, epoch_demes in demes_present.items():
        if len(epoch_demes) > 5:
            raise ValueError(
                f"Moments cannot integrate more than five demes at a time. "
                f"Epoch {epoch} has demes {epoch_demes}."
            )

    # get the list of size functions, migration matrices, and frozen attributes from
    # the deme graph and event times, matching the integration times
    nu_funcs, mig_mats, Ts, frozen_pops = _get_integration_parameters(
        g, demes_present, list_of_frozen_demes, Ne=Ne
    )

    # get the sample sizes within each deme, given sample sizes
    deme_sample_sizes = _get_deme_sample_sizes(
        g,
        demo_events,
        sampled_pops,
        sim_sample_sizes,
        demes_present,
        unsampled_n=unsampled_n,
    )

    # compute the SFS
    fs = _compute_sfs(
        demo_events,
        demes_present,
        deme_sample_sizes,
        nu_funcs,
        mig_mats,
        Ts,
        frozen_pops,
        theta=theta,
        gamma=gamma,
        h=h,
        reversible=reversible,
    )

    fs = _reorder_fs(fs, sampled_pops)

    # project down to desired sample sizes, if needed
    fs = fs.project(sample_sizes)
    # simplify pop id name if ancient sample at end time of that deme
    for ii, pid in enumerate(fs.pop_ids):
        if "_sampled_" in pid:
            p, t = pid.split("_sampled_")
            t = float(t.replace("_", "."))
            if t == sampled_deme_end_times[ii]:
                fs.pop_ids[ii] = p

    return fs


def LD(g, sampled_demes, sample_times=None, rho=None, theta=None, r=None, u=None):
    """
    Compute LD statistics from a ``demes``-specified demographic model.
    ``demes`` is a package for specifying demographic models in a
    user-friendly, human-readable YAML format. This function
    automatically parses the demographic model and returns LD statistics
    for the specified populations and (optionally) sampling times.

    No mutation or recombination rates are required. If recombination
    rates are omitted, only single-locus allelic diversity statistics
    will be computed. If mutation rates are omitted, that is, neither
    ``theta`` nor ``u`` are provided, the per-base mutation rate is set
    to 1. Thus, to recover the correctly scaled LD statistics, multiply
    the two-locus statistics by :math:`u^2` and the single-locus
    statistics by :math:`u`.

    :param g: A ``demes`` DemeGraph from which to compute the LD.
    :type g: :class:`demes.DemeGraph`
    :param sampled_demes: A list of deme IDs to take samples from. We can repeat
        demes, as long as the sampling of repeated deme IDs occurs at distinct
        times.
    :type sampled_demes: list of strings
    :param sample_times: If None, assumes all sampling occurs at the end of the
        existence of the sampled deme. If there are
        ancient samples, ``sample_times`` must be a list of same length as
        ``sampled_demes``, giving the sampling times for each sampled
        deme. Sampling times are given in time units of the original deme graph,
        so might not necessarily be generations (e.g. if ``g.time_units`` is years)
    :type sample_times: list of floats, optional
    :param rho: The population-size scaled recombination rate(s). Can be None, a
        non-negative float, or a list of values. Cannot be used with ``Ne``.
    :param theta: The population-size scaled mutation rate. Cannot be used
        with ``u``.
    :param r: The unscaled recombination rate(s). Can be None, a non-negative
        float, or a list of values. Recombination rates are scaled by ``Ne`` to
        get ``rho=4*Ne*u``, and ``Ne`` is determined by the root population size.
    :type r: scalar or list of scalars
    :param u: The raw per-base mutation rate. The reference effective population
        size ``Ne`` is determined from the demograhic model, after which
        ``theta`` is set to ``4*Ne*u``.
    :type u: scalar
    :return: A ``moments.LD`` LD statistics object, with number of populations equal
        to the length of ``sampled_demes``.
    :rtype: :class:`moments.LD.LDstats`
    """
    if sample_times is not None and len(sampled_demes) != len(sample_times):
        raise ValueError("sample_times must have same length as sampled_demes")
    for deme in sampled_demes:
        if deme not in g:
            raise ValueError(f"deme {deme} is not in demography")

    # we need to copy these to new names so they don't get updated during optimization
    deme_sample_times = copy.copy(sample_times)
    sampled_pops = copy.copy(sampled_demes)

    sampled_deme_end_times = [g[d].end_time for d in sampled_pops]
    if deme_sample_times is None:
        deme_sample_times = sampled_deme_end_times

    for t, d in zip(deme_sample_times, sampled_pops):
        if t < g[d].end_time or t >= g[d].start_time:
            raise ValueError(f"sample time {t} is outside of deme {d}'s time span")

    # for any ancient samples, we need to add frozen branches
    # with this, all "sample times" are at time 0, and ancient sampled demes are frozen
    if np.any(np.array(deme_sample_times) != 0):
        g, sampled_pops, list_of_frozen_demes = _augment_with_ancient_samples(
            g, sampled_pops, deme_sample_times
        )
        deme_sample_times = [0 for _ in deme_sample_times]
    else:
        list_of_frozen_demes = []

    if g.time_units != "generations":
        g, deme_sample_times = _convert_to_generations(g, deme_sample_times)
    for d, t in zip(sampled_pops, deme_sample_times):
        if t < g[d].end_time or t >= g[d].start_time:
            raise ValueError("sample time for {deme} must be within its time span")

    # get the list of demographic events from demes, which is a dictionary with
    # lists of splits, admixtures, mergers, branches, and pulses
    demes_demo_events = g.discrete_demographic_events()

    # get the dict of events and event times that partition integration epochs, in
    # descending order. events include demographic events, such as splits and
    # mergers and admixtures, as well as changes in population sizes or migration
    # rates that require instantaneous changes in the size function or migration matrix.
    # get the list of demes present in each epoch, as a dictionary with non-overlapping
    # adjoint epoch time intervals
    demo_events, demes_present = _get_demographic_events(
        g, demes_demo_events, sampled_pops
    )

    # get the list of size functions, migration matrices, and frozen attributes from
    # the deme graph and event times, matching the integration times
    Ne = _get_root_Ne(g)
    nu_funcs, mig_mats, Ts, frozen_pops = _get_integration_parameters(
        g, demes_present, list_of_frozen_demes, Ne=Ne
    )

    selfing_rates = _get_selfing_rates(g, demes_present)
    root_selfing_rate = _get_root_selfing_rate(g)

    # set recombination and mutation rates
    if rho is not None and r is not None:
        raise ValueError("Can only specify rho or r, but not both")
    if rho is None:
        if r is not None:
            if hasattr(r, "__len__"):
                rho = 4 * Ne * np.array(r)
            else:
                rho = 4 * Ne * r
    if rho is not None:
        if hasattr(rho, "__len__"):
            rho = np.array(rho)
            if np.any(rho < 0):
                raise ValueError("rho must be non-negative")
        else:
            if rho < 0:
                raise ValueError("rho must be non-negative")
    if u is None and theta is None:
        u = 1
    if u is not None:
        if theta is not None:
            raise ValueError("Only one of theta and u may be specified")
        theta = 4 * Ne * u
    if theta <= 0:
        raise ValueError("Mutation rate must be positive")

    # compute LD
    y = _compute_LD(
        demo_events,
        demes_present,
        nu_funcs,
        mig_mats,
        Ts,
        frozen_pops,
        selfing_rates,
        root_selfing_rate,
        rho,
        theta,
    )

    y = _reorder_LD(y, sampled_pops)

    # simplify pop id name if ancient sample at end time of that deme
    for ii, pid in enumerate(y.pop_ids):
        if "_sampled_" in pid:
            p, t = pid.split("_sampled_")
            t = float(t.replace("_", "."))
            if t == sampled_deme_end_times[ii]:
                y.pop_ids[ii] = p

    return y


def LDdecay(g, sampled_demes, rho=None, r=None, method="simpson", **kwargs):
    """
    Computes average LD statistics within recombination bins. The input demographic
    model, sampled demes, and other arguments follow ``moments.Demes.LD()``. Here,
    either ``rho`` or ``r`` must be given as an array of at least length 2.
    Recombination bins are defined by adjacent values in the list of recombination
    distances, so that there are ``len(r) - 1`` sets of LD statistics in the output
    ``LDstats`` object.

    Possible numerical integration methods:
      - "simpson", quadratic
      - "trapezoid", linear
      - "midpoint", zeroth order

    :param g: The input demographic model, loaded using ``demes``.
    :type g: :class:`demes.Graph`
    :param sampled_deme: List of deme names of demes in the input demographic
        model from which to draw samples from.
    :type sampled_deme: list of str
    :param rho: Monotonically increasing list of length two or more specifying
        recombination bin endpoints (in units of :math:`4 N_e r`). Only one of
        rho and r can be given.
    :type rho: list-like
    :param r: Monotonically increasing list of length two or more specifying
        recombination bin endpoints. Only one of rho and r can be given.
    :type r: list-like
    :param method: One of ``"simpson", ``"trapezoid"``, or ``"midpoint"``.
    :type method: str
    :param sample_times: If None, assumes all sampling occurs at the end of the
        existence of the sampled deme. If there are
        ancient samples, ``sample_times`` must be a list of same length as
        ``sampled_demes``, giving the sampling times for each sampled
        deme. Sampling times are given in time units of the original deme graph,
        so might not necessarily be generations (e.g. if ``g.time_units`` is years)
    :type sample_times: list of floats, optional
    :param theta: The population-size scaled mutation rate. Cannot be used
        with ``u``.
    :type theta: scalar
    :param u: The raw per-base mutation rate. The reference effective population
        size ``Ne`` is determined from the demograhic model, after which
        ``theta`` is set to ``4*Ne*u``.
    :type u: scalar
    :return: A ``moments.LD`` LD statistics object, with number of populations equal
        to the length of ``sampled_demes``.
    :rtype: :class:`moments.LD.LDstats`
    """
    if rho is not None:
        rho = np.asarray(rho)
        if not np.all(rho[1:] > rho[:-1]):
            raise ValueError("rho must be monotinically increasing")
    if r is not None:
        r = np.asarray(r)
        if not np.all(r[1:] > r[:-1]):
            raise ValueError("r must be monotinically increasing")

    if rho is not None and r is not None:
        raise ValueError("Only one of rho and r can be provided")
    if rho is None and r is None:
        raise ValueError(
            "Exactly one of rho and r must be given as a list of length at least two"
        )
    if rho is not None and len(rho) < 2:
        raise ValueError("rho must have length at least two")
    if r is not None and len(r) < 2:
        raise ValueError("r must have length at least two")

    possible_methods = ["simpson", "trapezoid", "midpoint"]
    if method not in possible_methods:
        raise ValueError(
            f"method {method} is not in possible methods: {possible_methods}"
        )

    if method == "simpson":
        if rho is not None:
            rho_pass = np.sort(np.concatenate((rho, (rho[1:] + rho[:-1]) / 2)))
            r_pass = None
        elif r is not None:
            r_pass = np.sort(np.concatenate((r, (r[1:] + r[:-1]) / 2)))
            rho_pass = None
    elif method == "trapezoid":
        if rho is not None:
            rho_pass = rho
            r_pass = None
        elif rs is not None:
            r_pass = r
            rho_pass = None
    else:
        assert method == "midpoint"
        if rho is not None:
            rho_pass = (rho[1:] + rho[:-1]) / 2
            r_pass = None
        elif r is not None:
            r_pass = (r[1:] + r[:-1]) / 2
            rho_pass = None

    # Additional kwargs that can be passed to LD() : sample_times=None, theta=None, u=None

    y = LD(g, sampled_demes, r=r_pass, rho=rho_pass, **kwargs)
    if method == "midpoint":
        return y
    elif method == "trapezoid":
        ld_binned = []
        for ld0, ld1 in zip(y.LD()[:-1], y.LD()[1:]):
            ld_binned.append((ld0 + ld1) / 2)
        y_new = moments.LD.LDstats(
            ld_binned + [y.H()], num_pops=y.num_pops, pop_ids=y.pop_ids
        )
        return y_new
    elif method == "simpson":
        ld_binned = []
        for i in range((len(y.LD()) - 1) // 2):
            ld0 = y.LD()[2 * i]
            ldc = y.LD()[2 * i + 1]
            ld1 = y.LD()[2 * i + 2]
            ld_binned.append((ld0 + 4 * ldc + ld1) / 6)
        y_new = moments.LD.LDstats(
            ld_binned + [y.H()], num_pops=y.num_pops, pop_ids=y.pop_ids
        )
        return y_new


##
## general functions used by both SFS and LD
##


def _convert_to_generations(g, sample_times):
    """
    Takes a deme graph that is not in time units of generations and converts
    times to generations, using the time units and generation times given.
    """
    if g.time_units == "generations":
        return g, sample_times
    else:
        for ii, sample_time in enumerate(sample_times):
            sample_times[ii] = sample_time / g.generation_time
        g = g.in_generations()
        return g, sample_times


def _augment_with_ancient_samples(g, sampled_demes, sample_times):
    """
    Returns a demography object and new sampled demes where we add
    a branch event for the new sampled deme that is frozen.

    If all sample times are > 0, we also slice the graph to remove the
    time interval that is more recent than the most recent sample time.

    New sampled, frozen demes are labeled "{deme}_sampled_{sample_time}".
    Note that we cannot have multiple ancient sampling events at the same
    time for the same deme (for additional samples at the same time, increase
    the sample size).
    """
    # Adjust the graph if all sample times are greater than 0
    t = min(sample_times)
    g_new = moments.Demes.DemesUtil.slice(g, min(sample_times))
    sample_times = [st - t for st in sample_times]
    # add frozen branches
    frozen_demes = []
    b = demes.Builder.fromdict(g_new.asdict())
    for ii, (sd, st) in enumerate(zip(sampled_demes, sample_times)):
        if st > 0 or t > 0:
            sd_frozen = sd + f"_sampled_{'_'.join(str(float(st + t)).split('.'))}"
            # update names of sampled demes
            sampled_demes[ii] = sd_frozen
            deme_sample_times = [
                y for x, y in zip(sampled_demes, sample_times) if x == sd
            ]
            if st > 0:
                # add the frozen branch, as sample time is nonzero
                frozen_demes.append(sd_frozen)
                b.add_deme(
                    sd_frozen,
                    start_time=st,
                    epochs=[dict(end_time=0, start_size=1)],
                    ancestors=[sd],
                )
            elif t > 0:
                # change the name of the sampled branch, as we have all ancient samples
                for ii, d in enumerate(b.data["demes"]):
                    if d["name"] == sd:
                        b.data["demes"][ii]["name"] = sd_frozen
                # change migration and pulse demes involving this sampled deme
                if "migrations" in b.data.keys():
                    for ii, m in enumerate(b.data["migrations"]):
                        if m["source"] == sd:
                            m["source"] = sd_frozen
                        if m["dest"] == sd:
                            m["dest"] = sd_frozen
                        b.data["migrations"][ii] = m
                if "pulses" in b.data.keys():
                    for ii, p in enumerate(b.data["pulses"]):
                        for jj, source in enumerate(p["sources"]):
                            if source == sd:
                                p["sources"][jj] = sd_frozen
                        if p["dest"] == sd:
                            p["dest"] = sd_frozen
                        b.data["pulses"][ii] = p
    g_new = b.resolve()
    return g_new, sampled_demes, frozen_demes


def _get_demographic_events(g, demes_demo_events, sampled_demes):
    """
    Returns demographic events and present demes over each epoch.
    Epochs are divided by any demographic event.
    """
    # first get set of all time dividers, from demographic events, migration
    # rate changes, deme epoch changes
    break_points = set()
    for deme in g.demes:
        for e in deme.epochs:
            break_points.add(e.start_time)
            break_points.add(e.end_time)
    for pulse in g.pulses:
        break_points.add(pulse.time)
    for migration in g.migrations:
        break_points.add(migration.start_time)
        break_points.add(migration.end_time)

    # get demes present for each integration epoch
    integration_times = [
        (start_time, end_time)
        for start_time, end_time in zip(
            sorted(list(break_points))[-1:0:-1], sorted(list(break_points))[-2::-1]
        )
    ]

    # find live demes in each epoch, starting with most ancient
    demes_present = defaultdict(list)
    # add demes as they appear from past to present to end of lists
    deme_start_times = defaultdict(list)
    for deme in g.demes:
        deme_start_times[deme.start_time].append(deme.name)

    if math.inf not in deme_start_times.keys():
        raise ValueError("Root deme must have start time as inf")
    if len(deme_start_times[math.inf]) != 1:
        raise ValueError("Deme graph can only have a single root")

    for start_time in sorted(deme_start_times.keys())[::-1]:
        for deme_id in deme_start_times[start_time]:
            end_time = g[deme_id].end_time
            for interval in integration_times:
                if start_time >= interval[0] and end_time <= interval[1]:
                    demes_present[interval].append(deme_id)

    # Dictionary of demographic events, occurring in the order:
    #   branches, pulses, admixtures, mergers, splits.
    # Importantly, splits and mergers remove the parental populations, so if
    # there are events like branches or pulses that involve those parental
    # populations at the same time, they will not be present when we try to
    # apply those events, resulting in an error.
    demo_events = defaultdict(list)
    for branch in demes_demo_events["branches"]:
        event = ("branch", branch.parent, branch.child)
        demo_events[branch.time].append(event)
    for pulse in demes_demo_events["pulses"]:
        event = ("pulse", pulse.sources, pulse.dest, pulse.proportions)
        demo_events[pulse.time].append(event)
    for admix in demes_demo_events["admixtures"]:
        event = ("admix", admix.parents, admix.proportions, admix.child)
        demo_events[admix.time].append(event)
    for merge in demes_demo_events["mergers"]:
        event = ("merge", merge.parents, merge.proportions, merge.child)
        demo_events[merge.time].append(event)
    for split in demes_demo_events["splits"]:
        event = ("split", split.parent, split.children)
        demo_events[split.time].append(event)

    # if there are any unsampled demes that end before present and do not have
    # any descendent demes, we need to add marginalization events.
    for deme_id, succs in g.successors().items():
        if deme_id not in sampled_demes and (
            len(succs) == 0
            or np.all([g[succ].start_time > g[deme_id].end_time for succ in succs])
        ):
            event = ("marginalize", deme_id)
            demo_events[g[deme_id].end_time].append(event)

    return demo_events, demes_present


def _get_root_Ne(g):
    # get root population and set Ne to root size
    for deme_id, preds in g.predecessors().items():
        if len(preds) == 0:
            root_deme = deme_id
            break
    Ne = g[root_deme].epochs[0].start_size
    return Ne


def _get_integration_parameters(g, demes_present, frozen_list, Ne=None):
    """
    Returns a list of size functions, migration matrices, integration times,
    and lists frozen demes.
    """
    nu_funcs = []
    integration_times = []
    migration_matrices = []
    frozen_demes = []

    if Ne is None:
        Ne = _get_root_Ne(g)
    else:
        if Ne != _get_root_Ne(g):
            warnings.warn(
                "Input Ne is different from root population initial size, "
                "subsequent population size scaling may be incorrect"
            )

    for interval, live_demes in sorted(demes_present.items())[::-1]:
        # get intergration time for interval
        T = (interval[0] - interval[1]) / 2 / Ne
        if T == math.inf:
            T = 0
        integration_times.append(T)
        # get frozen attributes
        freeze = [d in frozen_list for d in live_demes]
        frozen_demes.append(freeze)
        # get nu_function or list of sizes (if all constant)
        sizes = []
        for d in live_demes:
            sizes.append(_sizes_at_time(g, d, interval))
        nu_func = _make_nu_func(sizes, T, Ne)
        nu_funcs.append(nu_func)
        # get migration matrix for interval
        mig_mat = np.zeros((len(live_demes), len(live_demes)))
        for ii, d_from in enumerate(live_demes):
            for jj, d_to in enumerate(live_demes):
                if d_from != d_to:
                    m = _migration_rate_in_interval(g, d_from, d_to, interval)
                    mig_mat[jj, ii] = 2 * Ne * m
        migration_matrices.append(mig_mat)

    return nu_funcs, migration_matrices, integration_times, frozen_demes


def _make_nu_func(sizes, T, Ne):
    """
    Given the sizes at start and end of time interval, and the size function for
    each deme, along with the integration time and reference Ne, return the
    size function that gets passed to the moments integration routines.
    """
    if np.all([s[-1] == "constant" for s in sizes]):
        # all constant
        nu_func = [s[0] / Ne for s in sizes]
    else:
        nu_funcs_separated = []
        for s in sizes:
            if s[-1] == "constant":
                assert s[0] == s[1]
                nu_funcs_separated.append(lambda t, N0=s[0]: N0 / Ne)
            elif s[-1] == "linear":
                nu_funcs_separated.append(
                    lambda t, N0=s[0], NF=s[1]: N0 / Ne + t / T * (NF - N0) / Ne
                )
            elif s[-1] == "exponential":
                nu_funcs_separated.append(
                    lambda t, N0=s[0], NF=s[1]: N0
                    / Ne
                    * np.exp(np.log(NF / N0) * t / T)
                )
            else:
                raise ValueError(f"{s[-1]} not a valid size function")

        def nu_func(t):
            return [nu(t) for nu in nu_funcs_separated]

        # check that this is correct, or if we have to "pin" parameters
    return nu_func


def _sizes_at_time(g, deme_id, time_interval):
    """
    Returns the start size, end size, and size function for given deme over the
    given time interval.
    """
    for epoch in g[deme_id].epochs:
        if epoch.start_time >= time_interval[0] and epoch.end_time <= time_interval[1]:
            break
    if epoch.size_function not in ["constant", "exponential", "linear"]:
        raise ValueError(
            "Can only intergrate constant, exponential, or linear size functions"
        )
    size_function = epoch.size_function

    if size_function == "constant":
        start_size = end_size = epoch.start_size

    if epoch.start_time == time_interval[0]:
        start_size = epoch.start_size
    else:
        if size_function == "exponential":
            start_size = epoch.start_size * np.exp(
                np.log(epoch.end_size / epoch.start_size)
                * (epoch.start_time - time_interval[0])
                / epoch.time_span
            )
        elif size_function == "linear":
            frac = (epoch.start_time - time_interval[0]) / epoch.time_span
            start_size = epoch.start_size + frac * (epoch.end_size - epoch.start_size)

    if epoch.end_time == time_interval[1]:
        end_size = epoch.end_size
    else:
        if size_function == "exponential":
            end_size = epoch.start_size * np.exp(
                np.log(epoch.end_size / epoch.start_size)
                * (epoch.start_time - time_interval[1])
                / epoch.time_span
            )
        elif size_function == "linear":
            frac = (epoch.start_time - time_interval[1]) / epoch.time_span
            end_size = epoch.start_size + frac * (epoch.end_size - epoch.start_size)

    return start_size, end_size, size_function


def _migration_rate_in_interval(g, source, dest, time_interval):
    """
    Get the migration rate from source to dest over the given time interval.
    """
    rate = 0
    for mig in g.migrations:
        try:  # if asymmetric migration
            if mig.source == source and mig.dest == dest:
                if (
                    mig.start_time >= time_interval[0]
                    and mig.end_time <= time_interval[1]
                ):
                    rate = mig.rate
        except AttributeError:  # symmetric migration
            if source in mig.demes and dest in mig.demes:
                if (
                    mig.start_time >= time_interval[0]
                    and mig.end_time <= time_interval[1]
                ):
                    rate = mig.rate
    return rate


##
## Functions for SFS computation
##


def _get_deme_sample_sizes(
    g, demo_events, sampled_demes, sample_sizes, demes_present, unsampled_n=4
):
    """
    Returns sample sizes within each deme that is present within each interval.
    Deme samples sizes can change if there are pulse or branching events, e.g.,
    but will be constant over the integration epochs.
    This works by climbing up the demography from most recent integration epoch to
    most distant. Unsampled leaf demes get size unsampled_ns, and others have size
    given by sample_sizes.
    """
    ns = {}
    for interval, deme_ids in demes_present.items():
        ns[interval] = [0 for _ in deme_ids]

    # initialize with sampled demes and unsampled, marginalized demes
    for deme_id, n in zip(sampled_demes, sample_sizes):
        for interval in ns.keys():
            if interval[0] <= g[deme_id].start_time:
                ns[interval][demes_present[interval].index(deme_id)] += n

    # Climb up the demographic events, taking into account pulses, branches, etc
    # when we add a new deme, determine base n from its successors (split, merge,
    # admixture), and propagate up. Similarly, propagate up other events that add
    # lineages to a branch (branches, pulses). Marginalize events add the deme
    # sample size with unsampled_n.
    for t, events in sorted(demo_events.items()):
        for event in events[::-1]:
            if event[0] == "marginalize":
                deme_id = event[1]
                # add unsampled deme
                for interval in ns.keys():
                    if (
                        interval[0] <= g[deme_id].start_time
                        and interval[1] >= g[deme_id].end_time
                    ):
                        ns[interval][
                            demes_present[interval].index(deme_id)
                        ] += unsampled_n
            elif event[0] == "split":
                # add the parental deme
                deme_id = event[1]
                children = event[2]
                for interval in sorted(ns.keys()):
                    if interval[0] == g[deme_id].end_time:
                        # get child sizes at time of split
                        children_ns = {
                            child: ns[interval][demes_present[interval].index(child)]
                            for child in children
                        }
                    if (
                        interval[0] <= g[deme_id].start_time
                        and interval[1] >= g[deme_id].end_time
                    ):
                        for child in children:
                            ns[interval][
                                demes_present[interval].index(deme_id)
                            ] += children_ns[child]
            elif event[0] == "branch":
                # add child n to parent n for integration epochs above t
                deme_id = event[1]
                child = event[2]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        # get child sizes at time of split
                        child_ns = ns[interval][demes_present[interval].index(child)]
                    if (
                        interval[0] <= g[deme_id].start_time
                        and interval[1] >= g[deme_id].end_time
                        and interval[1] >= t
                    ):
                        ns[interval][demes_present[interval].index(deme_id)] += child_ns
            elif event[0] == "pulse":
                # figure out how much the admix_in_place needs from child to parent
                sources = event[1]
                dest = event[2]
                for source in sources:
                    for interval in sorted(ns.keys()):
                        if interval[1] == t:
                            dest_size = ns[interval][
                                demes_present[interval].index(dest)
                            ]
                        if (
                            interval[0] <= g[source].start_time
                            and interval[1] >= g[source].end_time
                            and interval[1] >= t
                        ):
                            ns[interval][
                                demes_present[interval].index(source)
                            ] += dest_size
            elif event[0] == "merge":
                # each parent gets number of lineages in child
                parents = event[1]
                child = event[3]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        child_size = ns[interval][demes_present[interval].index(child)]
                    for parent in parents:
                        if (
                            interval[0] <= g[parent].start_time
                            and interval[1] >= g[parent].end_time
                        ):
                            ns[interval][
                                demes_present[interval].index(parent)
                            ] += child_size
            elif event[0] == "admix":
                # each parent gets num child lineages for all epochs above t
                parents = event[1]
                child = event[3]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        child_size = ns[interval][demes_present[interval].index(child)]
                    for parent in parents:
                        if (
                            interval[0] <= g[parent].start_time
                            and interval[1] >= g[parent].end_time
                            and interval[1] >= t
                        ):
                            ns[interval][
                                demes_present[interval].index(parent)
                            ] += child_size
    return ns


def _set_up_selection_dicts(gamma, h):
    if type(gamma) is dict:
        gamma_dict = copy.copy(gamma)
        if "_default" not in gamma_dict:
            gamma_dict["_default"] = 0
    else:
        gamma_dict = {"_default": gamma}
    if h is None:
        h_dict = {"_default": 0.5}
    elif type(h) is dict:
        h_dict = copy.copy(h)
        if "_default" not in h_dict:
            h_dict["_default"] = 0.5
    else:
        h_dict = {"_default": h}
    return gamma_dict, h_dict


def _compute_sfs(
    demo_events,
    demes_present,
    deme_sample_sizes,
    nu_funcs,
    migration_matrices,
    integration_times,
    frozen_demes,
    theta=1.0,
    gamma=None,
    h=None,
    reversible=False,
):
    """
    Integrates using moments to find the SFS for given demo events, etc
    """
    if reversible is True:
        assert type(theta) is list
        assert len(theta) == 2
        # theta is forward and backward rates, as list of length 2
        theta_fd = theta[0]
        theta_bd = theta[1]
        assert theta_fd < 1 and theta_bd < 1
        mask_corners = False
    else:
        # theta is a scalar
        assert isinstance(theta, (int, float))
        mask_corners = True

    integration_intervals = sorted(list(demes_present.keys()))[::-1]
    root_deme = demes_present[integration_intervals[0]][0]

    # set up gamma and h as a dictionary covering all demes
    gamma_dict, h_dict = _set_up_selection_dicts(gamma, h)

    # set up initial steady-state 1D SFS for ancestral deme
    n0 = deme_sample_sizes[integration_intervals[0]][0]
    if gamma is None:
        gamma0 = 0.0
    else:
        if root_deme in gamma_dict:
            gamma0 = gamma_dict[root_deme]
        else:
            gamma0 = gamma_dict["_default"]
    if h is None:
        h0 = 0.5
    else:
        if root_deme in h_dict:
            h0 = h_dict[root_deme]
        else:
            h0 = h_dict["_default"]

    if reversible is False:
        fs = theta * moments.LinearSystem_1D.steady_state_1D(n0, gamma=gamma0, h=h0)
    else:
        fs = moments.LinearSystem_1D.steady_state_1D_reversible(
            n0, gamma=gamma0, theta_fd=theta_fd, theta_bd=theta_bd
        )
        if h0 != 0.5:
            raise ValueError("can only use h=0.5 with reversible mutation model")
    fs = moments.Spectrum(fs, pop_ids=[root_deme], mask_corners=mask_corners)

    # for each set of demographic events and integration epochs, step through
    # integration, apply events, and then reorder populations to align with demes
    # present in the next integration epoch
    for T, nu, M, frozen, interval in zip(
        integration_times,
        nu_funcs,
        migration_matrices,
        frozen_demes,
        integration_intervals,
    ):
        if T > 0:
            if gamma is not None:
                gamma_int = [
                    gamma_dict[pid] if pid in gamma_dict else gamma_dict["_default"]
                    for pid in fs.pop_ids
                ]
                h_int = [
                    h_dict[pid] if pid in h_dict else h_dict["_default"]
                    for pid in fs.pop_ids
                ]
            else:
                gamma_int = None
                h_int = None
            if reversible:
                fs.integrate(
                    nu,
                    T,
                    m=M,
                    frozen=frozen,
                    gamma=gamma_int,
                    h=h_int,
                    finite_genome=True,
                    theta_fd=theta_fd,
                    theta_bd=theta_bd,
                )
            else:
                fs.integrate(
                    nu, T, m=M, frozen=frozen, gamma=gamma_int, h=h_int, theta=theta
                )

        events = demo_events[interval[1]]
        for event in events:
            fs = _apply_event(fs, event, interval[1], deme_sample_sizes, demes_present)

        if interval[1] > 0:
            # rearrange to next order of demes
            next_interval = integration_intervals[
                [x[0] for x in integration_intervals].index(interval[1])
            ]
            next_deme_order = demes_present[next_interval]
            assert fs.ndim == len(next_deme_order)
            assert np.all([d in next_deme_order for d in fs.pop_ids])
            fs = _reorder_fs(fs, next_deme_order)

    return fs


def _apply_event(fs, event, t, deme_sample_sizes, demes_present):
    e = event[0]
    if e == "marginalize":
        fs = fs.marginalize([fs.pop_ids.index(event[1])])
    elif e == "split":
        children = event[2]
        if len(children) == 1:
            # "split" into just one population (name change)
            fs.pop_ids[fs.pop_ids.index(event[1])] = children[0]
        else:
            # split into multiple children demes
            if len(children) + len(fs.pop_ids) - 1 > 5:
                raise ValueError("Cannot apply split that creates more than 5 demes")
            # get children deme sizes at time t
            for i, ns in deme_sample_sizes.items():
                if i[0] == t:
                    split_sizes = [
                        deme_sample_sizes[i][demes_present[i].index(c)]
                        for c in children
                    ]
                    break
            split_idx = fs.pop_ids.index(event[1])
            # children[0] is placed in split idx, the rest are at the end
            fs = _split_fs(fs, split_idx, children, split_sizes)
    elif e == "branch":
        # use fs.branch function, new in 1.1.5
        parent = event[1]
        child = event[2]
        for i, ns in deme_sample_sizes.items():
            if i[0] == t:
                branch_size = deme_sample_sizes[i][demes_present[i].index(child)]
                break
        branch_idx = fs.pop_ids.index(parent)
        fs = fs.branch(branch_idx, branch_size, new_id=child)
    elif e in ["admix", "merge"]:
        # two or more populations merge, based on given proportions
        parents = event[1]
        proportions = event[2]
        child = event[3]
        for i, ns in deme_sample_sizes.items():
            if i[0] == t:
                child_size = deme_sample_sizes[i][demes_present[i].index(child)]
        fs = _admix_fs(fs, parents, proportions, child, child_size)
    elif e == "pulse":
        # admixture from one population to another, with some proportion
        sources = event[1]
        dest = event[2]
        proportions = event[3]
        fs = _pulse_fs(fs, sources, dest, proportions)
    else:
        raise ValueError(f"Haven't implemented methods for event type {e}")
    return fs


def _split_fs(fs, split_idx, children, split_sizes):
    """
    Split the SFS into children with split_sizes, from the deme at split_idx.
    """
    i = 1
    while i < len(children):
        fs = fs.split(
            split_idx,
            split_sizes[0] + sum(split_sizes[i + 1 :]),
            split_sizes[i],
            new_ids=[children[0], children[i]],
        )
        i += 1
    return fs


def _admix_fs(fs, parents, proportions, child, child_size):
    """
    Both merge and admixture events use this function, with the only difference that
    merge events remove the parental demes, while admixture events do not.
    """
    if len(parents) >= 2:
        # admix first two pops
        fA = proportions[0] / (proportions[0] + proportions[1])
        fB = proportions[1] / (proportions[0] + proportions[1])
        assert np.isclose(fA, 1 - fB)
        idxA = fs.pop_ids.index(parents[0])
        idxB = fs.pop_ids.index(parents[1])
        fs = fs.admix(idxA, idxB, child_size, fA, new_id=child)
    if len(parents) >= 3:
        # admix third pop
        fAB = (proportions[0] + proportions[1]) / (
            proportions[0] + proportions[1] + proportions[2]
        )
        fC = proportions[2] / (proportions[0] + proportions[1] + proportions[2])
        assert np.isclose(fAB, 1 - fC)
        idxAB = fs.pop_ids.index(child)  # last pop, was added to end
        idxC = fs.pop_ids.index(parents[2])
        fs = fs.admix(idxAB, idxC, child_size, fAB, new_id=child)
    if len(parents) >= 4:
        # admix 4th pop
        fABC = (proportions[0] + proportions[1] + proportions[2]) / (
            proportions[0] + proportions[1] + proportions[2] + proportions[3]
        )
        fD = proportions[3] / (
            proportions[0] + proportions[1] + proportions[2] + proportions[3]
        )
        assert np.isclose(fABC, 1 - fD)
        idxABC = fs.pop_ids.index(child)
        idxD = fs.pop_ids.index(parents[3])
        fs = fs.admix(idxABC, idxC, child_size, fABC, new_id=child)
    if len(parents) == 5:
        # admix 5th pop
        fABCD = (proportions[0] + proportions[1] + proportions[2] + proportions[3]) / (
            proportions[0]
            + proportions[1]
            + proportions[2]
            + proportions[3]
            + proportions[4]
        )
        fE = proportions[4] / (
            proportions[0]
            + proportions[1]
            + proportions[2]
            + proportions[3]
            + proportions[4]
        )
        assert np.isclose(fABCD, 1 - fE)
        idxABCD = fs.pop_ids.index(child)
        idxE = fs.pop_ids.index(parents[4])
        fs = fs.admix(idxABCD, idxE, child_size, fABCD, new_id=child)
    return fs


def _pulse_fs(fs, sources, dest, proportions):
    dest_idx = fs.pop_ids.index(dest)
    for ii, (source, proportion) in enumerate(zip(sources, proportions)):
        # uses admix in place
        source_idx = fs.pop_ids.index(source)
        # in the source deme, we keep that size minus the dest size
        keep_from = fs.sample_sizes[source_idx] - fs.sample_sizes[dest_idx]
        adjusted_proportion = proportion / (1 - sum(proportions[ii + 1 :]))
        fs = fs.pulse_migrate(source_idx, dest_idx, keep_from, adjusted_proportion)
    return fs


def _reorder_fs(fs, next_deme_order):
    """
    Takes a SFS with given population order and returns a SFS with the order
    from ``next_deme_order``. Uses ``fs.swap_axes(idx1, idx2)`` sequentially
    through populations.

    :param next_deme_order: List of population IDs in the order of output SFS.
    """
    if np.any([id not in fs.pop_ids for id in next_deme_order]) or np.any(
        [id not in next_deme_order for id in fs.pop_ids]
    ):
        raise ValueError("fs.pop_ids and next_deme_order have mismatched IDs")

    out = copy.deepcopy(fs)
    for ii, swap_id in enumerate(next_deme_order):
        pop_id = out.pop_ids[ii]
        if pop_id != swap_id:
            swap_index = out.pop_ids.index(swap_id)
            out = out.swap_axes(ii, swap_index)
    return out


##
## Functions for LD computation
##


def _get_selfing_rates(g, demes_present):
    """
    Returns a list of size functions, migration matrices, integration times,
    and lists frozen demes.
    """
    selfing_rates = []

    for interval, live_demes in sorted(demes_present.items())[::-1]:
        # get selfing_rates for interval
        interval_rates = []
        for d in live_demes:
            # get the selfing rate for deme d in epoch that spans this interval
            for epoch in g[d].epochs:
                if epoch.start_time >= interval[0] and epoch.end_time <= interval[1]:
                    interval_rates.append(epoch.selfing_rate)
        selfing_rates.append(interval_rates)

    return selfing_rates


def _get_root_selfing_rate(g):
    for deme_id, preds in g.predecessors().items():
        if len(preds) == 0:
            root_deme = deme_id
            break
    root_selfing_rate = g[root_deme].epochs[0].selfing_rate
    return root_selfing_rate


def _compute_LD(
    demo_events,
    demes_present,
    nu_funcs,
    migration_matrices,
    integration_times,
    frozen_demes,
    selfing_rates,
    root_selfing_rate,
    rho,
    theta,
):
    integration_intervals = sorted(list(demes_present.keys()))[::-1]

    # set up initial steady-state LD for ancestral deme
    y = moments.LD.LDstats(
        moments.LD.Numerics.steady_state(
            [1], theta=theta, rho=rho, selfing_rate=[root_selfing_rate]
        ),
        num_pops=1,
        pop_ids=demes_present[integration_intervals[0]],
    )

    # for each set of demographic events and integration epochs, step through
    # integration, apply events, and then reorder populations to align with demes
    # present in the next integration epoch
    for T, nu, M, frozen, interval, selfing_rate in zip(
        integration_times,
        nu_funcs,
        migration_matrices,
        frozen_demes,
        integration_intervals,
        selfing_rates,
    ):
        if np.all([s is None for s in selfing_rate]):
            selfing = None
        else:
            selfing = [s if s is not None else 0 for s in selfing_rate]

        if T > 0:
            y.integrate(
                nu, T, m=M, frozen=frozen, theta=theta, rho=rho, selfing=selfing
            )

        events = demo_events[interval[1]]
        for event in events:
            y = _apply_LD_event(y, event, interval[1], demes_present)

        if interval[1] > 0:
            # rearrange to next order of demes
            next_interval = integration_intervals[
                [x[0] for x in integration_intervals].index(interval[1])
            ]
            # marginalize populations that are not in next interval
            # possibly due to admixture event coinciding with deme end time
            next_deme_order = demes_present[next_interval]
            to_marginalize = [x for x in y.pop_ids if x not in next_deme_order]
            for marg_deme in to_marginalize:
                event = ("marginalize", marg_deme)
                y = _apply_LD_event(y, event, interval[1], demes_present)
            assert y.num_pops == len(next_deme_order)
            assert np.all([d in next_deme_order for d in y.pop_ids])
            y = _reorder_LD(y, next_deme_order)

    return y


def _apply_LD_event(y, event, t, demes_present):
    e = event[0]
    if e == "marginalize":
        y = y.marginalize([y.pop_ids.index(event[1])])
    elif e == "split":
        children = event[2]
        if len(children) == 1:
            # "split" into just one population (name change)
            y.pop_ids[y.pop_ids.index(event[1])] = children[0]
        else:
            split_idx = y.pop_ids.index(event[1])
            # children[0] is placed in split idx, the rest are at the end
            i = 1
            while i < len(children):
                y = y.split(split_idx, new_ids=[children[0], children[i]])
                i += 1
    elif e == "branch":
        # branch is a split, but keep the pop_id of parent
        parent = event[1]
        child = event[2]
        split_idx = y.pop_ids.index(parent)
        y = y.split(split_idx, new_ids=[parent, child])
    elif e in ["admix", "merge"]:
        # two or more populations merge, based on given proportions
        parents = event[1]
        proportions = event[2]
        child = event[3]
        # if e == "admix":
        #    marg = False
        # elif e == "merge":
        #    marg = True
        y = _admix_LD(y, parents, proportions, child, marginalize=False)
    elif e == "pulse":
        dest = event[2]
        dest_idx = y.pop_ids.index(dest)
        for ii, (source, proportion) in enumerate(zip(event[1], event[3])):
            # admixture from one or more populations to another, with some proportion
            source_idx = y.pop_ids.index(source)
            adjusted_proportion = proportion / (1 - sum(event[3][ii + 1 :]))
            y = y.pulse_migrate(source_idx, dest_idx, adjusted_proportion)
    else:
        raise ValueError(f"Haven't implemented methods for event type {e}")
    return y


def _admix_LD(y, parents, proportions, child, marginalize=False):
    """
    Both merge and admixture events use this function, with the only difference that
    merge events remove the parental demes, while admixture events do not.
    """
    if len(parents) >= 2:
        # admix first two pops
        fA = proportions[0] / (proportions[0] + proportions[1])
        fB = proportions[1] / (proportions[0] + proportions[1])
        assert np.isclose(fA, 1 - fB)
        idxA = y.pop_ids.index(parents[0])
        idxB = y.pop_ids.index(parents[1])
        y = y.admix(idxA, idxB, fA, new_id=child)
    if len(parents) >= 3:
        i = 2
        while i < len(parents):
            f = proportions[i] / sum(proportions[: i + 1])
            idx = y.pop_ids.index(parents[i])
            y = y.pulse_migrate(idx, y.num_pops - 1, f)
            i += 1
    if marginalize:
        marg_indexes = [y.pop_ids.index(p) for p in parents]
        y = y.marginalize(marg_indexes)
    return y


def _reorder_LD(y, next_deme_order):
    """
    :param y: LD stats object
    :param next_deme_order: List of population IDs in the order of output SFS.
    """
    ### this function could be combined with ``_reorder_fs`` if swap_pops has
    ### swap_axes as an alias...
    if np.any([id not in y.pop_ids for id in next_deme_order]) or np.any(
        [id not in next_deme_order for id in y.pop_ids]
    ):
        raise ValueError("y.pop_ids and next_deme_order have mismatched IDs")

    out = copy.copy(y)
    for ii, swap_id in enumerate(next_deme_order):
        pop_id = out.pop_ids[ii]
        if pop_id != swap_id:
            swap_index = out.pop_ids.index(swap_id)
            out = out.swap_pops(ii, swap_index)
    return out
