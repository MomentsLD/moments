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
        one_sided = [False] * len(p0)

    pwork = numpy.array(p0, copy=True, dtype=float)
    if ii == jj:
        if pwork[ii] != 0 and not one_sided[ii]:
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
        if (
            pwork[ii] != 0
            and pwork[jj] != 0
            and not one_sided[ii]
            and not one_sided[jj]
        ):
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


def _get_hess(func, p0, eps, args=()):
    """
    Calculate Hessian matrix of partial second derivatives.
    Hij = dfunc/(dp_i dp_j)

    :param func: Model function
    :param p0: Parameter values to take derivative around
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    one_sided = [False] * len(p0)
    for i, pval in enumerate(p0):
        if pval != 0:
            # Account for floating point arithmetic issues
            if pval * eps_in < 1e-12:
                eps[i] = eps_in
                one_sided[i] = True
            else:
                eps[i] = eps_in * pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    f0 = func(p0, *args)
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = _hessian_elem(
                func, f0, p0, ii, jj, eps, args=args, one_sided=one_sided
            )
            hess[jj][ii] = hess[ii][jj]
    return hess


def _get_grad(func, p0, eps, args=()):
    """
    Calculate gradient vector

    :param func: Model function
    :param p0: Parameters for func
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    one_sided = [False] * len(p0)
    for i, pval in enumerate(p0):
        if pval != 0:
            # Account for floating point arithmetic issues
            if pval * eps_in < 1e-12:
                eps[i] = eps_in
                one_sided[i] = True
            else:
                eps[i] = eps_in * pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    grad = numpy.empty([len(p0), 1])
    for ii in range(len(p0)):
        pwork = numpy.array(p0, copy=True, dtype=float)

        if p0[ii] != 0 and not one_sided[ii]:
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


_ld_cache = {}


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

    # Cache evaluations of the LDstats inside our hessian/J
    # evaluation function
    def func(params, m, v):
        key = tuple(params)
        if key not in _ld_cache:
            _ld_cache[key] = func_ex(params, statistics)
        y = _ld_cache[key]
        ll = Inference.ll_over_bins(y, m, v)
        return -ll

    def log_func(logparams, m, v):
        return func(numpy.exp(logparams), m, v)

    # First calculate the observed hessian.
    if not log:
        hess = -_get_hess(func, p0, eps, args=[ms, vcs])
    else:
        hess = -_get_hess(log_func, numpy.log(p0), eps, args=[ms, vcs])

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
            grad_temp = _get_grad(log_func, numpy.log(p0), eps, args=[bs_ms, vcs])
        J_temp = numpy.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = numpy.linalg.inv(J)
    godambe = numpy.dot(numpy.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


def _remove_normalized_data(statistics, normalization, means, varcovs, all_boot):
    # get indexes to remove
    pi2_idx_to_del = statistics[0].index("pi2_{0}_{0}_{0}_{0}".format(normalization))
    H_idx_to_del = statistics[1].index("H_{0}_{0}".format(normalization))
    # remove from means
    for ii in range(len(means) - 1):
        means[ii] = np.delete(means[ii], pi2_idx_to_del)
    means[-1] = np.delete(means[-1], H_idx_to_del)
    # remove from varcovs
    for ii in range(len(varcovs) - 1):
        varcovs[ii] = np.delete(varcovs[ii], pi2_idx_to_del, axis=0)
        varcovs[ii] = np.delete(varcovs[ii], pi2_idx_to_del, axis=1)
    if varcovs[-1].size > 1:
        varcovs[-1] = np.delete(varcovs[-1], H_idx_to_del, axis=0)
        varcovs[-1] = np.delete(varcovs[-1], H_idx_to_del, axis=1)
    else:
        # Single poopulation data has 1D H array
        varcovs[-1] = np.delete(varcovs[-1], H_idx_to_del)
    # remove from all_boot
    if len(all_boot) > 0:
        for jj, boot in enumerate(all_boot):
            for ii in range(len(boot) - 1):
                all_boot[jj][ii] = np.delete(boot[ii], pi2_idx_to_del)
            all_boot[jj][-1] = np.delete(boot[-1], H_idx_to_del)
    else:
        all_boot = []
    # remove from statistics lists
    statistics[0].pop(pi2_idx_to_del)
    statistics[1].pop(H_idx_to_del)
    return statistics, means, varcovs, all_boot


_func_calls = 0
_output_stream = sys.stdout


def _expected_number_of_calls(params):
    n = len(params)
    return 4 * n * (n - 1) // 2 + 2 * n + 1


def _get_statistics_and_remove_normalization(
    model_func, p0, means, varcovs, all_boots, normalization, pass_Ne
):
    if pass_Ne:
        y = model_func(p0)
    else:
        y = model_func(p0[:-1])
    statistics = y.names()
    if (
        len(means[0]) != len(statistics[0]) or len(means[-1]) != len(statistics[-1])
    ) or (
        len(all_boots) > 0
        and (
            len(all_boots[0][0]) != len(statistics[0])
            or len(all_boots[0][-1]) != len(statistics[-1])
        )
    ):
        raise ValueError(
            "If 'statistics' is not given, then means, varcovs, and "
            "all_boots must have consistent sizes and be equal to the "
            "number of stats for the number of populations in the model."
        )
    # returns (statistics, means, varcovs, all_boots)
    return _remove_normalized_data(statistics, normalization, means, varcovs, all_boots)


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

    :param model_func: Model function
    :param all_boot: List of bootstrap LD stat means [m0, m1, m2, ...]
    :param p0: Best-fit parameters for model_func, with inferred Ne in last entry of
        parameter list.
    :param ms: See below..
    :param vcs: Original means and covariances of statistics from data. If statistics
        are not give, we remove the normalizing statistics. Otherwise, these need
        to be pared down so that the normalizing statistics are removed.
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param log: If True, assume log-normal distribution of parameters. Returned values
        are then the standard deviations of the *logs* of the parameter values,
        which can be interpreted as relative parameter uncertainties.
    :param return_GIM: If true, also return the full GIM.
    :param r_edges: The bin edges for LD statistics.
    :param normalization: The index of the population that we normalized by.
    :param pass_Ne: If True, Ne is a parameter in the model function, and by convention
        is the last entry in the parameters list. If False, Ne is only used to scale
        recombination rates.
    :param statistics: Statistics that we have included given as a list of lists:
        [ld_stats, h_stats]. If statistics is not given, we assume all statistics
        are included except for the normalizing statistic in each bin
    :param verbose: If an integer greater than 0, prints updates
        of the number of function calls and tested parameters
        at intervals given by that spacing.
    :type verbose: int, optional
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

    if statistics is None:
        # get statistics and remove normalizing statistic - requires that all statistics
        # in the input means and varcovs are present, including the normalizing stats
        (
            statistics,
            means,
            varcovs,
            all_boots,
        ) = _get_statistics_and_remove_normalization(
            model_func, p0, means, varcovs, all_boots, normalization, pass_Ne
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

    :param model_func: Model function
    :param p0: Best-fit parameters for model_func, with inferred Ne in last entry of
        parameter list.
    :param ms: See below..
    :param vcs: Original means and covariances of statistics from data. If statistics
        are not give, we remove the normalizing statistics. Otherwise, these need
        to be pared down so that the normalizing statistics are removed.
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
        Note that if eps*param is < 1e-12, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations.
    :param log: If True, assume log-normal distribution of parameters. Returned values
        are then the standard deviations of the *logs* of the parameter values,
        which can be interpreted as relative parameter uncertainties.
    :param return_GIM: If true, also return the full GIM.
    :param r_edges: The bin edges for LD statistics.
    :param normalization: The index of the population that we normalized by.
    :param pass_Ne: If True, Ne is a parameter in the model function, and by convention
        is the last entry in the parameters list. If False, Ne is only used to scale
        recombination rates.
    :param statistics: Statistics that we have included given as a list of lists:
        [ld_stats, h_stats]. If statistics is not given, we assume all statistics
        are included except for the normalizing statistic in each bin
    :param verbose: If an integer greater than 0, prints updates
        of the number of function calls and tested parameters
        at intervals given by that spacing.
    :type verbose: int, optional
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

    if statistics is None:
        # get statistics and remove normalizing statistic - requires that all statistics
        # in the input means and varcovs are present, including the normalizing stats
        statistics, means, varcovs, _ = _get_statistics_and_remove_normalization(
            model_func, p0, means, varcovs, [], normalization, pass_Ne
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
    # NOTE: There appears to be some discrepancy with +/- when returning log-likelihoods,
    # which makes returned hessian matrix negative
    uncerts = numpy.sqrt(numpy.diag(numpy.linalg.inv(-H)))
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
    Following `moments.Godambe.LRT_adjust()`, this function performs
    first-order moment matching to adjust the test statistic for the likelihood
    ratio test. Given log-likelihoods for a full and nested model (complex and
    simpler model, resp), the adjustment factor scales the difference in
    log-likelihoods, which can then be used in a chi-squared test, as
    implemented in `moments.Godambe.sum_chi2_ppf()`.

    :param model_func:
    :type model_func:
    :param all_boot:
    :type all_boot:
    :param p0:
    :type p0:
    :param nested_indices:
    :type nested_indices:

    """
    # TODO: complete docstring
    means = copy.deepcopy(ms)
    varcovs = copy.deepcopy(vcs)
    all_boots = copy.deepcopy(all_boot)
    rs = np.array(r_edges)

    if statistics is None:
        # get statistics and remove normalizing statistic - requires that all statistics
        # in the input means and varcovs are present, including the normalizing stats
        (
            statistics,
            means,
            varcovs,
            all_boots,
        ) = _get_statistics_and_remove_normalization(
            model_func, p0, means, varcovs, all_boots, normalization, pass_Ne
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

    GIM, H, J, cU = _get_godambe(
        pass_func, all_boots, p0, means, varcovs, eps, statistics, log=False
    )

    adjust = len(nested_indices) / numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust
