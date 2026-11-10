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
import sys, os


def _step_sizes(p0, eps, log=False):
    """
    Step sizes for finite-difference derivatives, and whether each derivative is
    one-sided.

    :param p0: Parameter values to take derivatives around.
    :param eps: Fractional step size. If ``log`` is True, ``p0`` are log-parameters,
        and the step is ``eps`` for each of them, which changes each parameter by
        the same relative amount, with central differences.
    :param log: If True, ``p0`` are log-parameters.
    :return: The step sizes and a list of whether each derivative is one-sided.
        Without ``log``, parameters equal to zero, or so small that ``eps`` times
        the parameter is below 1e-12, use a step of ``eps`` and one-sided
        (forward) differences, so that they are not stepped below zero.
    """
    steps = numpy.empty(len(p0))
    one_sided = [False] * len(p0)
    for i, pval in enumerate(p0):
        if log:
            steps[i] = eps
        elif abs(pval) * eps < 1e-12:
            steps[i] = eps
            one_sided[i] = True
        else:
            steps[i] = eps * abs(pval)
    return steps, one_sided


def _hessian_elem(func, f0, p0, ii, jj, eps, args=(), one_sided=None):
    """
    Calculate element [ii][jj] of the Hessian matrix, a matrix
    of partial second derivatives w.r.t. to parameters ii and jj

    :param func: Model function
    :param f0: Evaluation of func at p0
    :param p0: Parameters for func
    :param eps: List of absolute step sizes to use for each parameter when taking
        finite differences.
    :param args: Additional arguments to func
    :param one_sided: Optionally, pass in a sequence of length p0 that determines
        whether a one-sided derivative will be used for each parameter.
    """
    # Note that we need to specify dtype=float, to avoid this being an integer
    # array which will silently fail when adding fractional eps.
    if one_sided is None:
        one_sided = [p == 0 for p in p0]

    pwork = numpy.array(p0, copy=True, dtype=float)
    if ii == jj:
        if not one_sided[ii]:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork, *args)

            element = (fp - 2 * f0 + fm) / eps[ii] ** 2
        else:
            pwork[ii] = p0[ii] + 2 * eps[ii]
            fpp = func(pwork, *args)

            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            element = (fpp - 2 * fp + f0) / eps[ii] ** 2
    else:
        if not one_sided[ii] and not one_sided[jj]:
            # f(xi + hi, xj + h)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fpp = func(pwork, *args)

            # f(xi + hi, xj - hj)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] - eps[jj]
            fpm = func(pwork, *args)

            # f(xi - hi, xj + hj)
            pwork[ii] = p0[ii] - eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fmp = func(pwork, *args)

            # f(xi - hi, xj - hj)
            pwork[ii] = p0[ii] - eps[ii]
            pwork[jj] = p0[jj] - eps[jj]
            fmm = func(pwork, *args)

            element = (fpp - fpm - fmp + fmm) / (4 * eps[ii] * eps[jj])
        else:
            # f(xi + hi, xj + h)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fpp = func(pwork, *args)

            # f(xi + hi, xj)
            pwork[ii] = p0[ii] + eps[ii]
            pwork[jj] = p0[jj]
            fpm = func(pwork, *args)

            # f(xi, xj + hj)
            pwork[ii] = p0[ii]
            pwork[jj] = p0[jj] + eps[jj]
            fmp = func(pwork, *args)

            element = (fpp - fpm - fmp + f0) / (eps[ii] * eps[jj])
    return element


def _get_hess(func, p0, eps, args=(), log=False):
    """
    Calculate Hessian matrix of partial second derivatives.
    Hij = dfunc/(dp_i dp_j)

    :param func: Model function
    :param p0: Parameter values to take derivative around
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
        (see ``_step_sizes``).
    :param args: Additional arguments to func
    :param log: If True, ``p0`` are log-parameters, and the step size is ``eps``.
    """
    eps, one_sided = _step_sizes(p0, eps, log=log)

    f0 = func(p0, *args)
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = _hessian_elem(
                func, f0, p0, ii, jj, eps, args=args, one_sided=one_sided
            )
            hess[jj][ii] = hess[ii][jj]
    return hess


def _get_grad(func, p0, eps, args=(), log=False):
    """
    Calculate gradient vector

    :param func: Model function
    :param p0: Parameters for func
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
        (see ``_step_sizes``).
    :param args: Additional arguments to func
    :param log: If True, ``p0`` are log-parameters, and the step size is ``eps``.
    """
    eps, one_sided = _step_sizes(p0, eps, log=log)

    grad = numpy.empty([len(p0), 1])
    for ii in range(len(p0)):
        pwork = numpy.array(p0, copy=True, dtype=float)

        if not one_sided[ii]:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork, *args)

            grad[ii] = (fp - fm) / (2 * eps[ii])
        else:
            # Do one-sided finite-difference
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            pwork[ii] = p0[ii]
            fm = func(pwork, *args)

            grad[ii] = (fp - fm) / eps[ii]
    return grad


def _get_godambe(
    func_ex, all_boot, p0, ms, vcs, eps, statistics, log=False, just_hess=False
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

    def log_func(logparams, m, v):
        return func(numpy.exp(logparams), m, v)

    # First calculate the observed hessian.
    if not log:
        hess = -_get_hess(func, p0, eps, args=[ms, vcs])
    else:
        hess = -_get_hess(log_func, numpy.log(p0), eps, args=[ms, vcs], log=True)

    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = numpy.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = numpy.zeros((len(p0), 1))
    for bs_ms in all_boot:
        # boot = LDstats(boot)
        if not log:
            grad_temp = _get_grad(func, p0, eps, args=[bs_ms, vcs])
        else:
            grad_temp = _get_grad(
                log_func, numpy.log(p0), eps, args=[bs_ms, vcs], log=True
            )
        J_temp = numpy.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = numpy.linalg.inv(J)
    godambe = numpy.dot(numpy.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


_func_calls = 0
_output_stream = sys.stdout


def _expected_number_of_calls(params):
    n = len(params)
    return 4 * n * (n - 1) // 2 + 2 * n + 1


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
            + str(_expected_number_of_calls(p0))
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
            + str(_expected_number_of_calls(p0))
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

    adjust = len(nested_indices) / numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust
