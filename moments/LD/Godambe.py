"""
Parameter uncertainties are computed using Godambe information, described in
Coffman et al, MBE (2016). doi: https://doi.org/10.1093/molbev/msv255

If you use moments.LD.Godambe to compute parameter uncertainties, please cite
that paper. This was first developed by Alec Coffman for computing uncertainties
from inferences performed with dadi, modified here to handle LD decay curves.
"""

import numpy as np, numpy
from moments.LD import Inference
from moments.LD.LDstats_mod import LDstats
import copy

# from moments.Misc import delayed_flush
from .. import Misc
from .. import Godambe as _sfs_godambe
import sys, os


def _get_godambe(
    func_ex,
    all_boot,
    p0,
    ms,
    vcs,
    eps,
    statistics,
    log=False,
    just_hess=False,
    side=None,
    scale=None,
):
    """
    Godambe information and Hessian matrices

    :param func_ex: Model function
    :param all_boot: List of bootstrap frequency spectra
    :param p0: Best-fit parameters for func_ex.
    :param ms: Original data of statistics means.
    :param vcs: Original data of statistics variance covariance matrices.
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param statistics:
    :param log: If True, calculate derivatives in terms of log-parameters
    :param just_hess: If True, only evaluate and return the Hessian matrix
    :param side, scale: Optional, as in ``moments.Godambe.get_hess``.
    """

    # Cache evaluations of the LDstats inside our hessian/J evaluation function,
    # which reuse parameter values. The cache is local to this call, since the
    # statistics also depend on the model in func_ex, which isn't part of the key.
    ld_cache = {}

    def func(params, m, v):
        key = tuple(params)
        if key not in ld_cache:
            ld_cache[key] = func_ex(params, statistics)
        y = ld_cache[key]
        return Inference.ll_over_bins(m, y, v)

    return _sfs_godambe._godambe_matrices(
        func,
        p0,
        [ms, vcs],
        all_boot,
        lambda bs_ms: [bs_ms, vcs],
        eps,
        log,
        just_hess,
        side=side,
        scale=scale,
    )


_func_calls = 0
_output_stream = sys.stdout


def _remove_normalizing_statistics(
    model_func, p0, pass_Ne, means, varcovs, all_boots, normalization, statistics
):
    """
    Returns statistics, means, varcovs, and bootstrap means with the normalizing
    statistics removed, using ``moments.LD.Inference._remove_normalizing_statistics``.
    If ``statistics`` is None, the number of populations is taken from the model.
    """
    num_pops = None
    if statistics is None:
        y = model_func(p0) if pass_Ne else model_func(p0[:-1])
        num_pops = y.num_pops
    return Inference._remove_normalizing_statistics(
        means,
        varcovs,
        bootstraps=all_boots,
        normalization=normalization,
        num_pops=num_pops,
        statistics=statistics,
    )


def GIM_uncert(
    model_func,
    all_boot,
    p0,
    ms,
    vcs,
    log=False,
    eps=0.01,
    return_GIM=False,
    r_edges=None,
    normalization=0,
    pass_Ne=False,
    statistics=None,
    verbose=0,
):
    """
    Parameter uncertainties from Godambe Information Matrix (GIM). If you use this
    method, please cite
    `Coffman et al., MBE (2016) <https://doi.org/10.1093/molbev/msv255>`_.

    Returns standard deviations of parameter values.

    :param model_func: The demographic model function, called as
        ``model_func(params, rho=rho)`` and returning an LDstats object.
    :param all_boot: List of bootstrap LD stat means [m0, m1, m2, ...], as returned
        by ``moments.LD.Parsing.get_bootstrap_sets``, with the same statistics as
        ``ms``.
    :param p0: Best-fit parameters for model_func, with inferred Ne in last entry of
        parameter list.
    :param ms: The data means, as returned by ``moments.LD.Parsing.bootstrap_data``,
        with the statistics given by ``statistics``.
    :param vcs: The data variance-covariance matrices, as returned by
        ``moments.LD.Parsing.bootstrap_data``, matching ``ms``.
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param log: If True, assume log-normal distribution of parameters. Returned values
        are then the standard deviations of the *logs* of the parameter values,
        which can be interpreted as relative parameter uncertainties.
    :param return_GIM: If True, also return the full GIM.
    :param r_edges: The bin edges for LD statistics.
    :param normalization: The index of the population that we normalized by.
    :param pass_Ne: If True, Ne is a parameter in the model function, and by convention
        is the last entry in the parameters list. If False, Ne is only used to scale
        recombination rates.
    :param statistics: The statistics in the data, as [ld_stats, h_stats] in the
        standard order of ``moments.LD.Util.moment_names(num_pops)``. Defaults to
        None, in which case the data must contain the full set of statistics for
        the number of populations in the model, including the normalizing
        statistics, as returned by the ``moments.LD.Parsing`` functions. If the
        data contain a subset of statistics (e.g. from
        ``moments.LD.Parsing.subset_data``), list exactly those statistics. The
        normalizing statistics are removed before computing likelihoods.
    :param verbose: If an integer greater than 0, prints updates
        of the number of function calls and tested parameters
        at intervals given by that spacing.
        Note that ``verbose=1`` prints a line for every function evaluation, which can
        be many lines per second, and displaying them in a visible terminal window can
        slow down the computation. Use a larger value (e.g. 10 to 100) to avoid this.
    :type verbose: int, optional
    :return: The standard deviations of the parameters, and if ``return_GIM`` is
        True, the GIM.
    :rtype: numpy.ndarray, or tuple of numpy.ndarray
    """
    means = copy.deepcopy(ms)
    varcovs = copy.deepcopy(vcs)
    all_boots = copy.deepcopy(all_boot)
    rs = np.array(r_edges)

    if verbose > 0:
        _output_stream.write(
            "Expected number of function calls: "
            + str(_sfs_godambe._expected_number_of_calls(p0))
            + os.linesep
        )
        Misc.delayed_flush(delay=0.5)

    # remove the normalizing statistics, and get the names of the remaining ones
    statistics, means, varcovs, all_boots = _remove_normalizing_statistics(
        model_func, p0, pass_Ne, means, varcovs, all_boots, normalization, statistics
    )

    def pass_func(params, statistics):
        global _func_calls
        _func_calls += 1

        rho = 4 * params[-1] * rs
        if pass_Ne:
            y = Inference.bin_stats(model_func, params, rho=rho)
        else:
            y = Inference.bin_stats(model_func, params[:-1], rho=rho)
        y = Inference.sigmaD2(y, normalization=normalization)
        y = Inference.remove_nonpresent_statistics(y, statistics)

        if (verbose > 0) and (_func_calls % verbose == 0):
            param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
            _output_stream.write("%-8i, %s%s" % (_func_calls, param_str, os.linesep))
            Misc.delayed_flush(delay=0.5)

        return y

    GIM, H, J, cU = _get_godambe(
        pass_func, all_boots, p0, means, varcovs, eps, statistics, log=log
    )

    uncerts = numpy.sqrt(numpy.diag(numpy.linalg.inv(GIM)))
    if not return_GIM:
        return uncerts
    else:
        return uncerts, GIM


def FIM_uncert(
    model_func,
    p0,
    ms,
    vcs,
    log=False,
    eps=0.01,
    r_edges=None,
    normalization=0,
    pass_Ne=False,
    statistics=None,
    verbose=0,
):
    """
    Parameter uncertainties from Fisher Information Matrix. This approach typically
    underestimates the size of the true confidence intervals, as it does not take
    into account linkage between loci that causes data to be non-independent.

    Returns standard deviations of parameter values.

    :param model_func: The demographic model function, called as
        ``model_func(params, rho=rho)`` and returning an LDstats object.
    :param p0: Best-fit parameters for model_func, with inferred Ne in last entry of
        parameter list.
    :param ms: The data means, as returned by ``moments.LD.Parsing.bootstrap_data``,
        with the statistics given by ``statistics``.
    :param vcs: The data variance-covariance matrices, as returned by
        ``moments.LD.Parsing.bootstrap_data``, matching ``ms``.
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param log: If True, assume log-normal distribution of parameters. Returned values
        are then the standard deviations of the *logs* of the parameter values,
        which can be interpreted as relative parameter uncertainties.
    :param r_edges: The bin edges for LD statistics.
    :param normalization: The index of the population that we normalized by.
    :param pass_Ne: If True, Ne is a parameter in the model function, and by convention
        is the last entry in the parameters list. If False, Ne is only used to scale
        recombination rates.
    :param statistics: The statistics in the data, as [ld_stats, h_stats] in the
        standard order of ``moments.LD.Util.moment_names(num_pops)``. Defaults to
        None, in which case the data must contain the full set of statistics for
        the number of populations in the model, including the normalizing
        statistics, as returned by the ``moments.LD.Parsing`` functions. If the
        data contain a subset of statistics (e.g. from
        ``moments.LD.Parsing.subset_data``), list exactly those statistics. The
        normalizing statistics are removed before computing likelihoods.
    :param verbose: If an integer greater than 0, prints updates
        of the number of function calls and tested parameters
        at intervals given by that spacing.
        Note that ``verbose=1`` prints a line for every function evaluation, which can
        be many lines per second, and displaying them in a visible terminal window can
        slow down the computation. Use a larger value (e.g. 10 to 100) to avoid this.
    :type verbose: int, optional
    :return: The standard deviations of the parameters.
    :rtype: numpy.ndarray
    """
    means = copy.deepcopy(ms)
    varcovs = copy.deepcopy(vcs)
    rs = np.array(r_edges)

    if verbose > 0:
        _output_stream.write(
            "Expected number of function calls: "
            + str(_sfs_godambe._expected_number_of_calls(p0))
            + os.linesep
        )
        Misc.delayed_flush(delay=0.5)

    # remove the normalizing statistics, and get the names of the remaining ones
    statistics, means, varcovs, _ = _remove_normalizing_statistics(
        model_func, p0, pass_Ne, means, varcovs, [], normalization, statistics
    )

    def pass_func(params, statistics):
        global _func_calls
        _func_calls += 1

        rho = 4 * params[-1] * rs
        if pass_Ne:
            y = Inference.bin_stats(model_func, params, rho=rho)
        else:
            y = Inference.bin_stats(model_func, params[:-1], rho=rho)
        y = Inference.sigmaD2(y, normalization=normalization)
        y = Inference.remove_nonpresent_statistics(y, statistics)

        if (verbose > 0) and (_func_calls % verbose == 0):
            param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
            _output_stream.write("%-8i, %s%s" % (_func_calls, param_str, os.linesep))
            Misc.delayed_flush(delay=0.5)

        return y

    H = _get_godambe(
        pass_func, 0, p0, means, varcovs, eps, statistics, log=log, just_hess=True
    )
    uncerts = numpy.sqrt(numpy.diag(numpy.linalg.inv(H)))
    return uncerts


def LRT_adjust(
    model_func,
    all_boot,
    p0,
    nested_indices,
    ms,
    vcs,
    eps=0.01,
    r_edges=None,
    normalization=0,
    pass_Ne=False,
    statistics=None,
    check_hessian=True,
):
    """
    Following ``moments.Godambe.LRT_adjust()``, this function performs
    first-order moment matching to adjust the test statistic for the likelihood
    ratio test. Given log-likelihoods for a full and nested model (complex and
    simpler model, resp), the adjustment factor ``adj`` scales the likelihood ratio
    test statistic, ``D_adj = adj * 2 * (ll_full - ll_nested)``, whose p-value is
    given by ``moments.Godambe.sum_chi2_ppf()``. Note that the LD optimization
    functions return the negative log-likelihood.

    :param model_func: The demographic model function for the complex (full) model,
        called as ``model_func(params, rho=rho)`` and returning an LDstats object.
    :param all_boot: List of bootstrap LD stat means [m0, m1, m2, ...], as returned
        by ``moments.LD.Parsing.get_bootstrap_sets``, with the same statistics as
        ``ms``.
    :param p0: Best-fit parameters for the simple (nested) model, given as
        parameters of the complex model with the nested parameters set to their
        fixed values, and with Ne in the last entry.
    :param nested_indices: List of the positions of the nested parameters in the
        complex model's parameters.
    :param ms: The data means, as returned by ``moments.LD.Parsing.bootstrap_data``,
        with the statistics given by ``statistics``.
    :param vcs: The data variance-covariance matrices, as returned by
        ``moments.LD.Parsing.bootstrap_data``, matching ``ms``.
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
    :param r_edges: The bin edges for LD statistics.
    :param normalization: The index of the population that we normalized by.
    :param pass_Ne: If True, Ne is a parameter in the model function, and by convention
        is the last entry in the parameters list. If False, Ne is only used to scale
        recombination rates.
    :param statistics: The statistics in the data, as [ld_stats, h_stats] in the
        standard order of ``moments.LD.Util.moment_names(num_pops)``. Defaults to
        None, in which case the data must contain the full set of statistics for
        the number of populations in the model, including the normalizing
        statistics, as returned by the ``moments.LD.Parsing`` functions. If the
        data contain a subset of statistics (e.g. from
        ``moments.LD.Parsing.subset_data``), list exactly those statistics. The
        normalizing statistics are removed before computing likelihoods.
    :param check_hessian: If True, also compute the Hessian of the complex model over
        all of its parameters at ``p0``, and warn if it is indefinite, in which case
        the adjustment is unreliable (see the documentation on likelihood ratio
        tests). This takes roughly ``2 * k**2`` extra model
        evaluations for ``k`` parameters. The adjustment factor is the same either
        way.
    :return: The adjustment factor for the likelihood ratio test statistic.
    :rtype: float
    """
    means = copy.deepcopy(ms)
    varcovs = copy.deepcopy(vcs)
    all_boots = copy.deepcopy(all_boot)
    rs = np.array(r_edges)

    # remove the normalizing statistics, and get the names of the remaining ones
    statistics, means, varcovs, all_boots = _remove_normalizing_statistics(
        model_func, p0, pass_Ne, means, varcovs, all_boots, normalization, statistics
    )

    def pass_func(params, statistics):
        rho = 4 * params[-1] * rs
        if pass_Ne:
            y = Inference.bin_stats(model_func, params, rho=rho)
        else:
            y = Inference.bin_stats(model_func, params[:-1], rho=rho)
        y = Inference.sigmaD2(y, normalization=normalization)
        y = Inference.remove_nonpresent_statistics(y, statistics)
        return y

    # We only take derivatives with respect to the nested parameters, which are
    # fixed in the simple model, as in moments.Godambe.LRT_adjust.
    def diff_func(diff_params, statistics):
        full_params = numpy.array(p0, copy=True, dtype=float)
        full_params[nested_indices] = diff_params
        return pass_func(full_params, statistics)

    p_nested = numpy.asarray(p0, dtype=float)[nested_indices]
    GIM, H, J, cU = _get_godambe(
        diff_func, all_boots, p_nested, means, varcovs, eps, statistics, log=False
    )

    labels = [f"p0[{i}]" for i in range(len(p0) - 1)] + ["Ne"]
    if check_hessian:
        H_full = _get_godambe(
            pass_func,
            0,
            numpy.asarray(p0, dtype=float),
            means,
            varcovs,
            eps,
            statistics,
            log=False,
            just_hess=True,
        )
        _sfs_godambe._check_nested_hessian(
            H_full, labels, "moments.LD.Godambe.LRT_adjust"
        )
    else:
        _sfs_godambe._check_nested_hessian(
            H, [labels[i] for i in nested_indices], "moments.LD.Godambe.LRT_adjust"
        )

    adjust = len(nested_indices) / numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust
