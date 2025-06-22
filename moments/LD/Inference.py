import numpy as np
import math
import os, sys

from moments.LD import Numerics
from moments.LD import Util

import copy
import moments

# from moments.Misc import delayed_flush
from .. import Misc
from moments.LD.LDstats_mod import LDstats

from scipy.special import gammaln
import scipy.optimize

_counter = 0


def sigmaD2(y, normalization=0):
    """
    Compute the :math:`\\sigma_D^2` statistics normalizing by the heterozygosities
    in a given population.

    :param y: The input data.
    :type y: :class:`LDstats` object
    :param normalization: The index of the normalizing population
        (normalized by pi2_i_i_i_i and H_i_i), default set to 0.
    :type normalization: int, optional
    """
    if normalization >= y.num_pops or normalization < 0:
        raise ValueError("Normalization index must be for a present population")

    out = LDstats(copy.deepcopy(y[:]), num_pops=y.num_pops, pop_ids=y.pop_ids)

    for i in range(len(y))[:-1]:
        out[i] /= y[i][y.names()[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))]
    out[-1] /= y[-1][y.names()[1].index("H_{0}_{0}".format(normalization))]

    return out


def bin_stats(model_func, params, rho=[], theta=0.001, spread=None, kwargs={}):
    """
    Computes LD statist for a given model function over bins defined by ``rho``.
    Here, ``rho`` gives the bin edges, and we assume no spaces between bins. That
    is, if the length of the input recombination rates is :math:`l`, the number of
    bins is :math:`l-1`.

    :param model_func: The model function that takes parameters in the form
        ``model_func(params, rho=rho, theta=theta, **kwargs)``.
    :param params: The parameters to evaluate the model at.
    :type params: list of floats
    :param rho: The scaled recombination rate bin edges.
    :type rho: list of floats
    :param theta: The mutation rate
    :type theta: float, optional
    :param spread: A list of length rho-1 (number of bins), where each entry is an
        array of length rho+1 (number of bins plus amount outside bin range to each
        side). Each array must sum to one.
    :type spread: list of arrays
    :param kwargs: Extra keyword arguments to pass to ``model_func``.
    """
    if len(rho) < 2:
        raise ValueError(
            "number of recombination rates (bin edges) must be greater than one"
        )
    rho_mids = (np.array(rho[:-1]) + np.array(rho[1:])) / 2
    y_edges = model_func(params, rho=rho, theta=theta, **kwargs)
    y_mids = model_func(params, rho=rho_mids, theta=theta, **kwargs)
    y = [
        1.0 / 6 * (y_edges[i] + y_edges[i + 1] + 4 * y_mids[i])
        for i in range(len(rho_mids))
    ]
    if spread is None:
        y.append(y_edges[-1])
        return LDstats(y, num_pops=y_edges.num_pops, pop_ids=y_edges.pop_ids)
    else:
        if len(spread) != len(rho) - 1:
            raise ValueError("spread must be length of bins")
        y_spread = []
        for distr in spread:
            if len(distr) != len(rho) + 1:
                raise ValueError(
                    "spread distr is not the correct length (len(bins) + 2)"
                )
            if not np.isclose(np.sum(distr), 1):
                raise ValueError("spread distributions must sum to one")
            y_spread.append(
                (distr[0] * y_edges[0] + distr[1:-1].dot(y) + distr[-1] * y_edges[-2])
            )
        y_spread.append(y_edges[-1])
        return LDstats(y_spread, num_pops=y_edges.num_pops, pop_ids=y_edges.pop_ids)


def remove_normalized_lds(y, normalization=0):
    """
    Returns LD statistics with the normalizing statistic removed.

    :param y: An LDstats object that has been normalized to get
        :math:`\\sigma_D^2`-formatted statistics.
    :type y: :class:`LDstats` object
    :param normalization: The index of the normalizing population.
    :type normalization: int
    """
    to_delete_ld = y.names()[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))
    to_delete_h = y.names()[1].index("H_{0}_{0}".format(normalization))
    for i in range(len(y) - 1):
        if len(y[i]) != len(y.names()[0]):
            raise ValueError("Unexpected number of LD stats in data")
        y[i] = np.delete(y[i], to_delete_ld)
    if len(y[-1]) != len(y.names()[1]):
        raise ValueError("Unexpected number of H stats in data")
    y[-1] = np.delete(y[-1], to_delete_h)
    return y


def remove_normalized_data(
    means, varcovs, normalization=0, num_pops=1, statistics=None
):
    """
    Returns data means and covariance matrices with the normalizing
    statistics removed.

    :param means: List of means normalized statistics, where each entry is the
        full set of statistics for a given recombination distance.
    :type means: list of arrays
    :param varcovs: List of the corresponding variance covariance matrices.
    :type varcovs: list of arrays
    :param normalization: The index of the normalizing population.
    :type normalization: int
    :param num_pops: The number of populations in the data set.
    :type num_pops: int
    """
    if len(means) != len(varcovs):
        raise ValueError("Different lengths of means and covariances")
    if statistics is None:
        stats = Util.moment_names(num_pops)
    else:
        stats = copy.copy(statistics)
    to_delete_ld = stats[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))
    to_delete_h = stats[1].index("H_{0}_{0}".format(normalization))
    ms = []
    vcs = []
    for i in range(len(means) - 1):
        if (
            len(means[i]) != len(stats[0])
            or varcovs[i].shape[0] != len(stats[0])
            or varcovs[i].shape[1] != len(stats[0])
        ):
            raise ValueError(
                "Data and statistics mismatch. Some statistics are missing "
                "or the incorrect number of populations was given."
            )
        ms.append(np.delete(means[i], to_delete_ld))
        vcs.append(
            np.delete(np.delete(varcovs[i], to_delete_ld, axis=0), to_delete_ld, axis=1)
        )
    ms.append(np.delete(means[-1], to_delete_h))
    # Single population data will have 1-D array for H
    if varcovs[-1].size > 1:
        vcs.append(
            np.delete(np.delete(varcovs[-1], to_delete_h, axis=0), to_delete_h, axis=1)
        )
    else:
        vcs.append(np.delete(varcovs[-1], to_delete_h))
    if statistics is None:
        return ms, vcs
    else:
        stats[0].pop(to_delete_ld)
        stats[1].pop(to_delete_h)
        return ms, vcs, stats


def remove_nonpresent_statistics(y, statistics=[[], []]):
    """
    Removes data not found in the given set of statistics.

    :param y: LD statistics.
    :type y: :class:`LDstats` object.
    :param statistics: A list of lists for two and one locus statistics to keep.
    """
    to_delete = [[], []]
    for j in range(2):
        for i, s in enumerate(y.names()[j]):
            if s not in statistics[j]:
                to_delete[j].append(i)
    for i in range(len(y) - 1):
        y[i] = np.delete(y[i], to_delete[0])
    y[-1] = np.delete(y[-1], to_delete[1])
    return y


def _multivariate_normal_pdf(x, mu, Sigma):
    p = len(x)
    return np.sqrt(np.linalg.det(Sigma) / (2 * math.pi) ** p) * np.exp(
        -1.0 / 2 * np.dot(np.dot((x - mu).transpose(), np.linalg.inv(Sigma)), x - mu)
    )


def _ll(x, mu, Sigma_inv):
    """
    x = data
    mu = model function output
    Sigma_inv = inverse of the variance-covariance matrix
    """
    if len(x) == 0:
        return 0
    else:
        return -1.0 / 2 * np.dot(np.dot((x - mu).transpose(), Sigma_inv), x - mu)
        # - len(x)*np.pi - 1./2*np.log(np.linalg.det(Sigma))


_varcov_inv_cache = {}


def ll_over_bins(xs, mus, Sigmas):
    """
    Compute the composite log-likelihood over LD and heterozygosity statistics, given
    data and expectations. Inputs must be in the same order, and we assume each bin
    is independent, so we sum _ll(x, mu, Sigma) over each bin.

    :param xs: A list of data arrays.
    :param mus: A list of model function output arrays, same length as ``xs``.
    :param Sigmas: A list of var-cov matrices, same length as ``xs``.
    """
    it = iter([xs, mus, Sigmas])
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError(
            "Lists of data, means, and varcov matrices must be the same length"
        )
    ll_vals = []
    for ii in range(len(xs)):
        # get var-cov inverse from cache dictionary, or compute it
        recompute = True
        if (
            ii in _varcov_inv_cache
            and (_varcov_inv_cache[ii]["data"].size == Sigmas[ii].size)
            and np.all(_varcov_inv_cache[ii]["data"] == Sigmas[ii])
        ):
            Sigma_inv = _varcov_inv_cache[ii]["inv"]
            recompute = False
        if recompute:
            _varcov_inv_cache[ii] = {}
            _varcov_inv_cache[ii]["data"] = Sigmas[ii]
            if Sigmas[ii].size == 0:
                Sigma_inv = np.array([])
            else:
                Sigma_inv = np.linalg.inv(Sigmas[ii])
            _varcov_inv_cache[ii]["inv"] = Sigma_inv
        # append log-likelihood for this bin
        ll_vals.append(_ll(xs[ii], mus[ii], Sigma_inv))
    # sum over bins to get composite log-likelihood
    ll_val = np.sum(ll_vals)
    return ll_val


_out_of_bounds_val = -1e12


def _object_func(
    params,
    model_func,
    means,
    varcovs,
    fs=None,
    rs=None,
    theta=None,
    u=None,
    Ne=None,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0,
    normalization=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    use_afs=False,
    Leff=None,
    multinom=True,
    ns=None,
    statistics=None,
    pass_Ne=False,
    spread=None,
    output_stream=sys.stdout,
):
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    # Check our parameter bounds
    if lower_bound is not None:
        for pval, bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val
    if upper_bound is not None:
        for pval, bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val

    all_args = [params_up] + list(func_args)

    if theta is None:
        if Ne is None:
            Ne = params_up[-1]
            theta = 4 * Ne * u
            rhos = [4 * Ne * r for r in rs]
            if pass_Ne == False:
                all_args = [all_args[0][:-1]]
            else:
                all_args = [all_args[0][:]]
        else:
            theta = 4 * Ne * u
            rhos = [4 * Ne * r for r in rs]
    else:
        if Ne is not None:
            rhos = [4 * Ne * r for r in rs]

    ## first get ll of afs
    if use_afs == True:
        if Leff is None:
            model = theta * model_func[1](all_args[0], ns)
        else:
            model = Leff * theta * model_func[1](all_args[0], ns)
        if fs.folded:
            model = model.fold()
        if multinom == True:
            ll_afs = moments.Inference.ll_multinom(model, fs)
        else:
            ll_afs = moments.Inference.ll(model, fs)

    ## next get ll for LD stats
    func_kwargs = {"theta": theta, "rho": rhos, "spread": spread}
    stats = bin_stats(model_func[0], *all_args, **func_kwargs)
    stats = sigmaD2(stats, normalization=normalization)
    if statistics == None:
        stats = remove_normalized_lds(stats, normalization=normalization)
    else:
        stats = remove_nonpresent_statistics(stats, statistics=statistics)
    simp_stats = stats[:-1]
    het_stats = stats[-1]

    if use_afs == False:
        simp_stats.append(het_stats)

    ## resulting ll from afs (if used) plus ll from rho bins
    if use_afs == True:
        result = ll_afs + ll_over_bins(means, simp_stats, varcovs)
    else:
        result = ll_over_bins(means, simp_stats, varcovs)

    # Bad result
    if np.isnan(result):
        print("got bad results...")
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params_up]))
        output_stream.write(
            "%-8i, %-12g, %s%s" % (_counter, result, param_str, os.linesep)
        )
        Misc.delayed_flush(delay=flush_delay)

    return -result


def _object_func_log(log_params, *args, **kwargs):
    return _object_func(np.exp(log_params), *args, **kwargs)


def optimize_log_fmin(
    p0,
    data,
    model_func,
    rs=None,
    theta=None,
    u=2e-8,
    Ne=None,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    normalization=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    use_afs=False,
    Leff=None,
    multinom=False,
    ns=None,
    statistics=None,
    pass_Ne=False,
    spread=None,
    maxiter=None,
    maxfun=None,
):
    """
    Optimize (using the log of) the parameters using a downhill simplex
    algorithm. Initial parameters ``p0``, the data ``[means, varcovs]``,
    the demographic ``model_func``, and ``rs`` to specify recombination
    bin edges are required. ``Ne`` must either be specified as a keyword
    argument or is included as the *last* parameter in ``p0``.

    :param p0: The initial guess for demographic parameters,
        demography parameters plus (optionally) Ne.
    :type p0: list
    :param data: The parsed data[means, varcovs, fs]. The frequency spectrum
        fs is optional, and used only if use_afs=True.

        - Means: The list of mean statistics within each bin
          (has length ``len(rs)`` or ``len(rs) - 1`` if using AFS). If we are
          not using the AFS, which is typical, the heterozygosity statistics
          come last.
        - varcovs: The list of varcov matrices matching the data in ``means``.

    :type data: list
    :param model_func: The demographic model to compute statistics
        for a given rho. If we are using AFS, it's a list of the two models
        [LD func, AFS func]. If we're using LD stats alone, we pass a single LD
        model  as a list: [LD func].
    :type model_func: list
    :param rs: The list of raw recombination rates defining bin edges.
    :type rs: list
    :param theta: The population scaled per base mutation rate
        (4*Ne*mu, not 4*Ne*mu*L).
    :type theta: float, optional
    :param u: The raw per base mutation rate.
        Cannot be used with ``theta``.
    :type u: float, optional
    :param Ne: The fixed effective population size to scale
        u and r. If ``Ne`` is a parameter to fit, it should be the last parameter
        in ``p0``.
    :type Ne: float, optional
    :param lower_bound: Defaults to ``None``. Constraints on the
        lower bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type lower_bound: list, optional
    :param upper_bound: Defaults to ``None``. Constraints on the
        upper bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type upper_bound: list, optional
    :param verbose: If an integer greater than 0, prints updates
        of the optimization procedure at intervals given by that spacing.
    :type verbose: int, optional
    :param func_args: Additional arguments to be passed
        to ``model_func``.
    :type func_args: list, optional
    :param func_kwargs: Additional keyword arguments to be
        passed to ``model_func``.
    :type func_kwargs: dict, optional
    :param fixed_params: Defaults to ``None``. To fix some
        parameters, this should be a list of equal length as ``p0``, with
        ``None`` for parameters to be fit and fixed values at corresponding
        indexes.
    :type fixed_params: list, optional
    :param use_afs: Defaults to ``False``. We can pass a model
        to compute the frequency spectrum and use
        that instead of heterozygosity statistics for single-locus data.
    :type use_afs: bool, optional
    :param Leff: The effective length of genome from which
        the fs was generated (only used if fitting to afs).
    :type Leff: float, optional
    :param multinom: Only used if we are fitting the AFS.
        If ``True``, the likelihood is computed for an optimally rescaled FS.
        If ``False``, the likelihood is computed for a fixed scaling of the FS
        found by theta=4*Ne*u and Leff
    :type multinom: bool, optional
    :param ns: The sample size, which is only needed
        if we are using the frequency spectrum, as the sample size does not
        affect mean LD statistics.
    :type ns: list of ints, optional
    :param statistics: Defaults to ``None``, which assumes that
        all statistics are present and in the conventional default order. If
        the data is missing some statistics, we must specify which statistics
        are present using the subset of statistic names given by
        ``moments.LD.Util.moment_names(num_pops)``.
    :type statistics: list, optional
    :param pass_Ne: Defaults to ``False``. If ``True``, the
        demographic model includes ``Ne`` as a parameter (in the final position
        of input parameters).
    :type pass_Ne: bool, optional
    :param maxiter: Defaults to None. Maximum number of iterations to perform.
    :type maxiter: int
    :param maxfun: Defaults to None. Maximum number of function evaluations to make.
    :type maxfun: int
    """
    output_stream = sys.stdout

    means = data[0]
    varcovs = data[1]
    if use_afs is True:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError(
                "if use_afs=True, need to pass frequency spectrum, "
                "as data=[means,varcovs,fs]"
            )

        if ns is None:
            raise ValueError("need to set ns if we are fitting frequency spectrum")

    else:
        fs = None

    if rs is None:
        raise ValueError("need to pass rs as bin edges")

    # get num_pops
    if Ne is None:
        if not pass_Ne:
            y = model_func[0](p0[:-1])
        else:
            y = model_func[0](p0[:])
    else:
        y = model_func[0](p0)
    num_pops = y.num_pops

    # remove normalized statisticsd
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    if statistics is None:
        # if statistics is not None, assume we already filtered out the data
        ms, vcs = remove_normalized_data(
            ms, vcs, normalization=normalization, num_pops=num_pops
        )

    args = (
        model_func,
        ms,
        vcs,
        fs,
        rs,
        theta,
        u,
        Ne,
        lower_bound,
        upper_bound,
        verbose,
        flush_delay,
        normalization,
        func_args,
        func_kwargs,
        fixed_params,
        use_afs,
        Leff,
        multinom,
        ns,
        statistics,
        pass_Ne,
        spread,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin(
        _object_func_log,
        np.log(p0),
        args=args,
        full_output=True,
        disp=False,
        maxiter=maxiter,
        maxfun=maxfun,
    )

    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(np.exp(xopt), fixed_params)

    return xopt, fopt


def optimize_log_powell(
    p0,
    data,
    model_func,
    rs=None,
    theta=None,
    u=2e-8,
    Ne=None,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    normalization=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    use_afs=False,
    Leff=None,
    multinom=False,
    ns=None,
    statistics=None,
    pass_Ne=False,
    spread=None,
    maxiter=None,
    maxfun=None,
):
    """
    Optimize (using the log of) the parameters using the modified Powell's
    method, which optimizes slices of parameter space sequentially. Initial
    parameters ``p0``, the data ``[means, varcovs]``,
    the demographic ``model_func``, and ``rs`` to specify recombination
    bin edges are required. ``Ne`` must either be specified as a keyword
    argument or is included as the *last* parameter in ``p0``.

    :param p0: The initial guess for demographic parameters,
        demography parameters plus (optionally) Ne.
    :type p0: list
    :param data: The parsed data[means, varcovs, fs]. The frequency spectrum
        fs is optional, and used only if use_afs=True.

        - Means: The list of mean statistics within each bin
          (has length ``len(rs)`` or ``len(rs) - 1`` if using AFS). If we are
          not using the AFS, which is typical, the heterozygosity statistics
          come last.
        - varcovs: The list of varcov matrices matching the data in ``means``.

    :type data: list
    :param model_func: The demographic model to compute statistics
        for a given rho. If we are using AFS, it's a list of the two models
        [LD func, AFS func]. If we're using LD stats alone, we pass a single LD
        model  as a list: [LD func].
    :type model_func: list
    :param rs: The list of raw recombination rates defining bin edges.
    :type rs: list
    :param theta: The population scaled per base mutation rate
        (4*Ne*mu, not 4*Ne*mu*L).
    :type theta: float, optional
    :param u: The raw per base mutation rate.
        Cannot be used with ``theta``.
    :type u: float, optional
    :param Ne: The fixed effective population size to scale
        u and r. If ``Ne`` is a parameter to fit, it should be the last parameter
        in ``p0``.
    :type Ne: float, optional
    :param lower_bound: Defaults to ``None``. Constraints on the
        lower bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type lower_bound: list, optional
    :param upper_bound: Defaults to ``None``. Constraints on the
        upper bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type upper_bound: list, optional
    :param verbose: If an integer greater than 0, prints updates
        of the optimization procedure at intervals given by that spacing.
    :type verbose: int, optional
    :param func_args: Additional arguments to be passed
        to ``model_func``.
    :type func_args: list, optional
    :param func_kwargs: Additional keyword arguments to be
        passed to ``model_func``.
    :type func_kwargs: dict, optional
    :param fixed_params: Defaults to ``None``. To fix some
        parameters, this should be a list of equal length as ``p0``, with
        ``None`` for parameters to be fit and fixed values at corresponding
        indexes.
    :type fixed_params: list, optional
    :param use_afs: Defaults to ``False``. We can pass a model
        to compute the frequency spectrum and use
        that instead of heterozygosity statistics for single-locus data.
    :type use_afs: bool, optional
    :param Leff: The effective length of genome from which
        the fs was generated (only used if fitting to afs).
    :type Leff: float, optional
    :param multinom: Only used if we are fitting the AFS.
        If ``True``, the likelihood is computed for an optimally rescaled FS.
        If ``False``, the likelihood is computed for a fixed scaling of the FS
        found by theta=4*Ne*u and Leff
    :type multinom: bool, optional
    :param ns: The sample size, which is only needed
        if we are using the frequency spectrum, as the sample size does not
        affect mean LD statistics.
    :type ns: list of ints, optional
    :param statistics: Defaults to ``None``, which assumes that
        all statistics are present and in the conventional default order. If
        the data is missing some statistics, we must specify which statistics
        are present using the subset of statistic names given by
        ``moments.LD.Util.moment_names(num_pops)``.
    :type statistics: list, optional
    :param pass_Ne: Defaults to ``False``. If ``True``, the
        demographic model includes ``Ne`` as a parameter (in the final position
        of input parameters).
    :type pass_Ne: bool, optional
    :param maxiter: Defaults to None. Maximum number of iterations to perform.
    :type maxiter: int
    :param maxfun: Defaults to None. Maximum number of function evaluations to make.
    :type maxfun: int
    """
    output_stream = sys.stdout

    means = data[0]
    varcovs = data[1]
    if use_afs:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError(
                "if use_afs=True, need to pass frequency spectrum in data=[means,varcovs,fs]"
            )

        if ns is None:
            raise ValueError("need to set ns if we are fitting frequency spectrum")

    else:
        fs = None

    if rs is None:
        raise ValueError("need to pass rs as bin edges")

    # get num_pops
    if Ne is None:
        if not pass_Ne:
            y = model_func[0](p0[:-1])
        else:
            y = model_func[0](p0[:])
    else:
        y = model_func[0](p0)
    num_pops = y.num_pops

    # remove normalized statistics
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    if statistics is None:
        # if statistics is not None, assume we already filtered out the data
        ms, vcs = remove_normalized_data(
            ms, vcs, normalization=normalization, num_pops=num_pops
        )

    args = (
        model_func,
        ms,
        vcs,
        fs,
        rs,
        theta,
        u,
        Ne,
        lower_bound,
        upper_bound,
        verbose,
        flush_delay,
        normalization,
        func_args,
        func_kwargs,
        fixed_params,
        use_afs,
        Leff,
        multinom,
        ns,
        statistics,
        pass_Ne,
        spread,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_powell(
        _object_func_log,
        np.log(p0),
        args=args,
        full_output=True,
        disp=False,
        maxiter=maxiter,
        maxfun=maxfun,
    )

    xopt, fopt, direc, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(np.exp(xopt), fixed_params)

    return xopt, fopt


def optimize_log_lbfgsb(
    p0,
    data,
    model_func,
    rs=None,
    theta=None,
    u=2e-8,
    Ne=None,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    normalization=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    use_afs=False,
    Leff=None,
    multinom=False,
    ns=None,
    statistics=None,
    pass_Ne=False,
    spread=None,
    maxiter=40000,
    epsilon=1e-3,
    pgtol=1e-5,
):
    """
    Optimize (using the log of) the parameters using the modified Powell's
    method, which optimizes slices of parameter space sequentially. Initial
    parameters ``p0``, the data ``[means, varcovs]``,
    the demographic ``model_func``, and ``rs`` to specify recombination
    bin edges are required. ``Ne`` must either be specified as a keyword
    argument or is included as the *last* parameter in ``p0``.

    It is best at burrowing down a single minimum. This method is
    better than optimize_log if the optimum lies at one or more of the
    parameter bounds. However, if your optimum is not on the bounds, this
    method may be much slower.

    Because this works in log(params), it cannot explore values of params < 0.
    It should also perform better when parameters range over scales.

    The L-BFGS-B method was developed by Ciyou Zhu, Richard Byrd, and Jorge
    Nocedal. The algorithm is described in:

    - R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
      Constrained Optimization, (1995), SIAM Journal on Scientific and
      Statistical Computing , 16, 5, pp. 1190-1208.
    - C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
      FORTRAN routines for large scale bound constrained optimization (1997),
      ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550-560.

    :param p0: The initial guess for demographic parameters,
        demography parameters plus (optionally) Ne.
    :type p0: list
    :param data: The parsed data[means, varcovs, fs]. The frequency spectrum
        fs is optional, and used only if use_afs=True.

        - Means: The list of mean statistics within each bin
          (has length ``len(rs)`` or ``len(rs) - 1`` if using AFS). If we are
          not using the AFS, which is typical, the heterozygosity statistics
          come last.
        - varcovs: The list of varcov matrices matching the data in ``means``.

    :type data: list
    :param model_func: The demographic model to compute statistics
        for a given rho. If we are using AFS, it's a list of the two models
        [LD func, AFS func]. If we're using LD stats alone, we pass a single LD
        model  as a list: [LD func].
    :type model_func: list
    :param rs: The list of raw recombination rates defining bin edges.
    :type rs: list
    :param theta: The population scaled per base mutation rate
        (4*Ne*mu, not 4*Ne*mu*L).
    :type theta: float, optional
    :param u: The raw per base mutation rate.
        Cannot be used with ``theta``.
    :type u: float, optional
    :param Ne: The fixed effective population size to scale
        u and r. If ``Ne`` is a parameter to fit, it should be the last parameter
        in ``p0``.
    :type Ne: float, optional
    :param lower_bound: Defaults to ``None``. Constraints on the
        lower bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type lower_bound: list, optional
    :param upper_bound: Defaults to ``None``. Constraints on the
        upper bounds during optimization. These are given as lists of the same
        length of the parameters.
    :type upper_bound: list, optional
    :param verbose: If an integer greater than 0, prints updates
        of the optimization procedure at intervals given by that spacing.
    :type verbose: int, optional
    :param func_args: Additional arguments to be passed
        to ``model_func``.
    :type func_args: list, optional
    :param func_kwargs: Additional keyword arguments to be
        passed to ``model_func``.
    :type func_kwargs: dict, optional
    :param fixed_params: Defaults to ``None``. To fix some
        parameters, this should be a list of equal length as ``p0``, with
        ``None`` for parameters to be fit and fixed values at corresponding
        indexes.
    :type fixed_params: list, optional
    :param use_afs: Defaults to ``False``. We can pass a model
        to compute the frequency spectrum and use
        that instead of heterozygosity statistics for single-locus data.
    :type use_afs: bool, optional
    :param Leff: The effective length of genome from which
        the fs was generated (only used if fitting to afs).
    :type Leff: float, optional
    :param multinom: Only used if we are fitting the AFS.
        If ``True``, the likelihood is computed for an optimally rescaled FS.
        If ``False``, the likelihood is computed for a fixed scaling of the FS
        found by theta=4*Ne*u and Leff
    :type multinom: bool, optional
    :param ns: The sample size, which is only needed
        if we are using the frequency spectrum, as the sample size does not
        affect mean LD statistics.
    :type ns: list of ints, optional
    :param statistics: Defaults to ``None``, which assumes that
        all statistics are present and in the conventional default order. If
        the data is missing some statistics, we must specify which statistics
        are present using the subset of statistic names given by
        ``moments.LD.Util.moment_names(num_pops)``.
    :type statistics: list, optional
    :param pass_Ne: Defaults to ``False``. If ``True``, the
        demographic model includes ``Ne`` as a parameter (in the final position
        of input parameters).
    :type pass_Ne: bool, optional
    :param maxiter: Defaults to 40,000. Maximum number of iterations to perform.
    :type maxiter: int
    :param epsilon: Step-size to use for finite-difference derivatives.
    :type pgtol: float
    :param pgtol: Convergence criterion for optimization. For more info,
        see help(scipy.optimize.fmin_l_bfgs_b)
    :type pgtol: float
    """
    output_stream = sys.stdout

    means = data[0]
    varcovs = data[1]
    if use_afs:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError(
                "if use_afs=True, need to pass frequency spectrum in data=[means,varcovs,fs]"
            )

        if ns is None:
            raise ValueError("need to set ns if we are fitting frequency spectrum")

    else:
        fs = None

    if rs is None:
        raise ValueError("need to pass rs as bin edges")

    # get num_pops
    if Ne is None:
        if not pass_Ne:
            y = model_func[0](p0[:-1])
        else:
            y = model_func[0](p0[:])
    else:
        y = model_func[0](p0)
    num_pops = y.num_pops

    # remove normalized statistics
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    if statistics is None:
        # if statistics is not None, assume we already filtered out the data
        ms, vcs = remove_normalized_data(
            ms, vcs, normalization=normalization, num_pops=num_pops
        )

    args = (
        model_func,
        ms,
        vcs,
        fs,
        rs,
        theta,
        u,
        Ne,
        lower_bound,
        upper_bound,
        verbose,
        flush_delay,
        normalization,
        func_args,
        func_kwargs,
        fixed_params,
        use_afs,
        Leff,
        multinom,
        ns,
        statistics,
        pass_Ne,
        spread,
        output_stream,
    )

    # Make bounds list. For this method it needs to be in terms of log params.
    if lower_bound is None:
        lower_bound = [None] * len(p0)
    else:
        lower_bound = np.log(lower_bound)
        lower_bound[np.isnan(lower_bound)] = None
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [None] * len(p0)
    else:
        upper_bound = np.log(upper_bound)
        upper_bound[np.isnan(upper_bound)] = None
    upper_bound = _project_params_down(upper_bound, fixed_params)
    bounds = list(zip(lower_bound, upper_bound))

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_l_bfgs_b(
        _object_func_log,
        np.log(p0),
        bounds=bounds,
        epsilon=epsilon,
        args=args,
        iprint=-1,
        pgtol=pgtol,
        maxiter=maxiter,
        approx_grad=True,
    )

    xopt, fopt, info_dict = outputs
    xopt = _project_params_up(np.exp(xopt), fixed_params)

    return xopt, fopt


def _project_params_down(pin, fixed_params):
    """
    Eliminate fixed parameters from pin.
    """
    if fixed_params is None:
        return pin

    if len(pin) != len(fixed_params):
        raise ValueError(
            "fixed_params list must have same length as input " "parameter array."
        )

    pout = []
    for ii, (curr_val, fixed_val) in enumerate(zip(pin, fixed_params)):
        if fixed_val is None:
            pout.append(curr_val)

    return np.array(pout)


def _project_params_up(pin, fixed_params):
    """
    Fold fixed parameters into pin.
    """
    if fixed_params is None:
        return pin

    if np.isscalar(pin):
        pin = [pin]

    pout = np.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout
