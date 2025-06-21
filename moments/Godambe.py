"""
Parameter uncertainties and likelihood ratio tests using Godambe information.
"""

import numpy

from . import Inference
from .Spectrum_mod import Spectrum


def _hessian_elem(func, f0, p0, ii, jj, eps, args=()):
    """
    Calculate element [ii][jj] of the Hessian matrix, a matrix
    of partial second derivatives w.r.t. to parameters ii and jj

    func: Model function
    f0: Evaluation of func at p0
    p0: Parameters for func
    eps: List of absolute step sizes to use for each parameter when taking
         finite differences.
    args: Additional arguments to func
    """
    # Note that we need to specify dtype=float, to avoid this being an integer
    # array which will silently fail when adding fractional eps.
    pwork = numpy.array(p0, copy=True, dtype=float)
    if ii == jj:
        if pwork[ii] != 0:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork, *args)

            element = (fp - 2 * f0 + fm) / eps[ii] ** 2
        if pwork[ii] == 0:
            pwork[ii] = p0[ii] + 2 * eps[ii]
            fpp = func(pwork, *args)

            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            element = (fpp - 2 * fp + f0) / eps[ii] ** 2
    else:
        if pwork[ii] != 0 and pwork[jj] != 0:
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


def get_hess(func, p0, eps, args=()):
    """
    Calculate Hessian matrix of partial second derivatives.
    Hij = dfunc/(dp_i dp_j)

    func: Model function
    p0: Parameter values to take derivative around
    eps: Fractional stepsize to use when taking finite-difference derivatives
    args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    for i, pval in enumerate(p0):
        if pval != 0:
            eps[i] = eps_in * pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    f0 = func(p0, *args)
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = _hessian_elem(func, f0, p0, ii, jj, eps, args=args)
            hess[jj][ii] = hess[ii][jj]
    return hess


def _get_grad(func, p0, eps, args=()):
    """
    Calculate gradient vector

    func: Model function
    p0: Parameters for func
    eps: Fractional stepsize to use when taking finite-difference derivatives
    args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    for i, pval in enumerate(p0):
        if pval != 0:
            eps[i] = eps_in * pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    grad = numpy.empty([len(p0), 1])
    for ii in range(len(p0)):
        pwork = numpy.array(p0, copy=True, dtype=float)

        if p0[ii] != 0:
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


def _get_godambe(func_ex, all_boot, p0, data, eps, log=False, just_hess=False):
    """
    Godambe information and Hessian matrices

    NOTE: Assumes that last parameter in p0 is theta.

    func_ex: Model function
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for func_ex.
    data: Original data frequency spectrum
    eps: Fractional stepsize to use when taking finite-difference derivatives
    log: If True, calculate derivatives in terms of log-parameters
    just_hess: If True, only evaluate and return the Hessian matrix
    """
    ns = data.sample_sizes

    # Cache evaluations of the frequency spectrum inside our hessian/J
    # evaluation function
    cache = {}

    def func(params, data):
        key = (tuple(params), tuple(ns))
        if key not in cache:
            cache[key] = func_ex(params, ns)
        fs = cache[key]
        return Inference.ll(fs, data)

    def log_func(logparams, data):
        return func(numpy.exp(logparams), data)

    # First calculate the observed hessian
    if not log:
        hess = -get_hess(func, p0, eps, args=[data])
    else:
        hess = -get_hess(log_func, numpy.log(p0), eps, args=[data])

    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = numpy.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = numpy.zeros((len(p0), 1))
    for ii, boot in enumerate(all_boot):
        boot = Spectrum(boot)
        if not log:
            grad_temp = _get_grad(func, p0, eps, args=[boot])
        else:
            grad_temp = _get_grad(log_func, numpy.log(p0), eps, args=[boot])

        J_temp = numpy.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = numpy.linalg.inv(J)
    godambe = numpy.dot(numpy.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


def GIM_uncert(
    func_ex, all_boot, p0, data, log=False, multinom=True, eps=0.01, return_GIM=False
):
    """
    Parameter uncertainties from Godambe Information Matrix (GIM).
    Returns standard deviations of parameter values. Bootstrap data
    is typically generated by splitting the genome into *N* chunks and
    sampling with replacement from those chunks *N* times.

    :param func_ex: Model function
    :type func_ex: demographic model
    :param all_boot: List of bootstrap frequency spectra
    :type all_boot: list of spectra
    :param p0: Best-fit parameters for func_ex
    :type p0: list-like
    :param data: Original data frequency spectrum
    :type data: spectrum object
    :param log: If True, assume log-normal distribution of parameters.
        Returned values are then the standard deviations of the *logs*
        of the parameter values, which can be interpreted as relative
        parameter uncertainties.
    :type log: bool
    :param multinom: If True, assume model is defined without an explicit
        parameter for theta. Because uncertainty in theta must be accounted
        for to get correct uncertainties for other parameters, this function
        will automatically consider theta if multinom=True. In that case, the
        final entry of the returned uncertainties will correspond to
        theta.
    :type multinom: bool
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
    :type eps: float
    :param return_GIM: If True, also return the full GIM.
    :type return_GIm: bool
    """
    if multinom:
        func_multi = func_ex
        model = func_multi(p0, data.sample_sizes)
        theta_opt = Inference.optimal_sfs_scaling(model, data)
        p0 = list(p0) + [theta_opt]
        func_ex = lambda p, ns: p[-1] * func_multi(p[:-1], ns)
    GIM, H, J, cU = _get_godambe(func_ex, all_boot, p0, data, eps, log)
    uncerts = numpy.sqrt(numpy.diag(numpy.linalg.inv(GIM)))
    if not return_GIM:
        return uncerts
    else:
        return uncerts, GIM


def FIM_uncert(func_ex, p0, data, log=False, multinom=True, eps=0.01):
    """
    Parameter uncertainties from Fisher Information Matrix.
    Returns standard deviations of parameter values.

    :param func_ex: Model function
    :type func_ex: demographic model
    :param p0: Best-fit parameters for func_ex
    :type p0: list-like
    :param data: Original data frequency spectrum
    :type data: spectrum object
    :param log: If True, assume log-normal distribution of parameters.
        Returned values are then the standard deviations of the *logs*
        of the parameter values, which can be interpreted as relative
        parameter uncertainties.
    :type log: bool
    :param multinom: If True, assume model is defined without an explicit
        parameter for theta. Because uncertainty in theta must be accounted
        for to get correct uncertainties for other parameters, this function
        will automatically consider theta if multinom=True. In that case, the
        final entry of the returned uncertainties will correspond to theta.
    :type multinom: bool
    :param eps: Fractional stepsize to use when taking
        finite-difference derivatives
    :type eps: float
    """
    if multinom:
        func_multi = func_ex
        model = func_multi(p0, data.sample_sizes)
        theta_opt = Inference.optimal_sfs_scaling(model, data)
        p0 = list(p0) + [theta_opt]
        func_ex = lambda p, ns: p[-1] * func_multi(p[:-1], ns)
    H = _get_godambe(func_ex, [], p0, data, eps, log, just_hess=True)
    return numpy.sqrt(numpy.diag(numpy.linalg.inv(H)))


def LRT_adjust(func_ex, all_boot, p0, data, nested_indices, multinom=True, eps=0.01):
    """
    First-order moment matching adjustment factor for likelihood ratio test.

    :param func_ex: Model function for complex model
    :type func_ex: demographic model
    :param all_boot: List of bootstrap frequency spectra
    :type all_boot: list of spectra
    :param p0: Best-fit parameters for the simple model, with nested parameter
        explicity defined.  Although equal to values for simple model, should
        be in a list form that can be taken in by the complex model you'd like
        to evaluate.
    :type p0: list-like
    :param data: Original data frequency spectrum
    :type data: spectrum object
    :param nested_indices: List of positions of nested parameters in complex model
        parameter list
    :type nested_indices: list of ints
    :param multinom: If True, assume model is defined without an explicit parameter
        for theta. Because uncertainty in theta must be accounted for to get
        correct uncertainties for other parameters, this function will
        automatically consider theta if multinom=True.
    :type multinom: bool
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
    :type eps: float
    """
    if multinom:
        func_multi = func_ex
        model = func_multi(p0, data.sample_sizes)
        theta_opt = Inference.optimal_sfs_scaling(model, data)
        p0 = list(p0) + [theta_opt]
        func_ex = lambda p, ns: p[-1] * func_multi(p[:-1], ns)

    # We only need to take derivatives with respect to the parameters in the
    # complex model that have been set to specified values in the simple model
    def diff_func(diff_params, ns):
        # diff_params argument is only the nested parameters. All the rest
        # should come from p0
        full_params = numpy.array(p0, copy=True, dtype=float)
        # Use numpy indexing to set relevant parameters
        full_params[nested_indices] = diff_params
        return func_ex(full_params, ns)

    p_nested = numpy.asarray(p0)[nested_indices]
    GIM, H, J, cU = _get_godambe(diff_func, all_boot, p_nested, data, eps, log=False)

    adjust = len(nested_indices) / numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust


def sum_chi2_ppf(x, weights=(0, 1)):
    """
    Percent point function (inverse of cdf) of weighted sum of chi^2
    distributions.

    x: Value(s) at which to evaluate ppf
    weights: Weights of chi^2 distributions, beginning with zero d.o.f.
             For example, weights=(0,1) is the normal chi^2 distribution with 1
             d.o.f. For single parameters on the boundary, the correct
             distribution for the LRT is 0.5*chi^2_0 + 0.5*chi^2_1, which would
             be weights=(0.5,0.5).
    """
    import scipy.stats.distributions as ssd

    # Ensure that weights are valid
    if abs(numpy.sum(weights) - 1) > 1e-6:
        raise ValueError("Weights must sum to 1.")
    # A little clunky, but we want to handle x = 0.5, and x = [2, 3, 4]
    # correctly. So if x is a scalar, we record that fact so we can return a
    # scalar on output.
    if numpy.isscalar(x):
        scalar_input = True
    else:
        scalar_input = False
    # Convert x into an array, so we can index it easily.
    x = numpy.atleast_1d(x)
    # Calculate total cdf of all chi^2 dists with dof > 1.
    # (ssd.chi2.cdf(x,0) is always nan, so we avoid that.)
    cdf = numpy.sum(
        [w * ssd.chi2.cdf(x, d + 1) for (d, w) in enumerate(weights[1:])], axis=0
    )
    # Add in contribution from 0 d.o.f.
    cdf[x > 0] += weights[0]
    # Convert to ppf
    ppf = 1 - cdf

    if scalar_input:
        return ppf[0]
    else:
        return ppf


def Wald_stat(
    func_ex,
    all_boot,
    p0,
    data,
    nested_indices,
    full_params,
    multinom=True,
    eps=0.01,
    adj_and_org=False,
):
    """
    Calculate test stastic from wald test

    func_ex: Model function for complex model
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for the simple model, with nested parameter
        explicity defined.  Although equal to values for simple model, should
        be in a list form that can be taken in by the complex model you'd like
        to evaluate.
    data: Original data frequency spectrum
    nested_indices: List of positions of nested parameters in complex model
    parameter list
    full_params: Parameter values for parameters found only in complex model,
                 Can either be array with just values found only in the compelx
                 model, or entire list of parameters from complex model.
    multinom: If True, assume model is defined without an explicit parameter for
              theta. Because uncertainty in theta must be accounted for to get
              correct uncertainties for other parameters, this function will
              automatically consider theta if multinom=True. In that case, the
              final entry of the returned uncertainties will correspond to
              theta.
    eps: Fractional stepsize to use when taking finite-difference derivatives
    adj_and_org: If False, return only adjusted Wald statistic. If True, also
                 return unadjusted statistic as second return value.
    """
    if multinom:
        func_multi = func_ex
        model = func_multi(p0, data.sample_sizes)
        theta_opt = Inference.optimal_sfs_scaling(model, data)
        # Also need to extend full_params
        if len(full_params) == len(p0):
            full_params = numpy.concatenate((full_params, [theta_opt]))
        p0 = list(p0) + [theta_opt]
        func_ex = lambda p, ns: p[-1] * func_multi(p[:-1], ns)

    # We only need to take derivatives with respect to the parameters in the
    # complex model that have been set to specified values in the simple model
    def diff_func(diff_params, ns):
        # diff_params argument is only the nested parameters. All the rest
        # should come from p0
        full_params = numpy.array(p0, copy=True, dtype=float)
        # Use numpy indexing to set relevant parameters
        full_params[nested_indices] = diff_params
        return func_ex(full_params, ns)

    # Reduce full params list to be same length as nested indices
    if len(full_params) == len(p0):
        full_params = numpy.asarray(full_params)[nested_indices]
    if len(full_params) != len(nested_indices):
        raise KeyError("Full parameters not equal in length to p0 or nested " "indices")

    p_nested = numpy.asarray(p0)[nested_indices]
    GIM, H, J, cU = _get_godambe(diff_func, all_boot, p_nested, data, eps, log=False)
    param_diff = full_params - p_nested

    wald_adj = numpy.dot(numpy.dot(numpy.transpose(param_diff), GIM), param_diff)
    wald_org = numpy.dot(numpy.dot(numpy.transpose(param_diff), H), param_diff)

    if adj_and_org:
        return wald_adj, wald_org
    return wald_adj


def score_stat(
    func_ex,
    all_boot,
    p0,
    data,
    nested_indices,
    multinom=True,
    eps=0.01,
    adj_and_org=False,
):
    """
    Calculate test stastic from score test

    func_ex: Model function for complex model
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for the simple model, with nested parameter
        explicity defined.  Although equal to values for simple model, should
        be in a list form that can be taken in by the complex model you'd like
        to evaluate.
    data: Original data frequency spectrum
    nested_indices: List of positions of nested parameters in complex model
                    parameter list
    eps: Fractional stepsize to use when taking finite-difference derivatives
    multinom: If True, assume model is defined without an explicit parameter for
              theta. Because uncertainty in theta must be accounted for to get
              correct uncertainties for other parameters, this function will
              automatically consider theta if multinom=True.
    adj_and_org: If False, return only adjusted score statistic. If True, also
                 return unadjusted statistic as second return value.
    """
    if multinom:
        func_multi = func_ex
        model = func_multi(p0, data.sample_sizes)
        theta_opt = Inference.optimal_sfs_scaling(model, data)
        p0 = list(p0) + [theta_opt]
        func_ex = lambda p, ns: p[-1] * func_multi(p[:-1], ns)

    # We only need to take derivatives with respect to the parameters in the
    # complex model that have been set to specified values in the simple model
    def diff_func(diff_params, ns):
        # diff_params argument is only the nested parameters. All the rest
        # should come from p0
        full_params = numpy.array(p0, copy=True, dtype=float)
        # Use numpy indexing to set relevant parameters
        full_params[nested_indices] = diff_params
        return func_ex(full_params, ns)

    p_nested = numpy.asarray(p0)[nested_indices]
    GIM, H, J, cU = _get_godambe(diff_func, all_boot, p_nested, data, eps, log=False)

    score_org = numpy.dot(numpy.dot(numpy.transpose(cU), numpy.linalg.inv(H)), cU)[0, 0]
    score_adj = numpy.dot(numpy.dot(numpy.transpose(cU), numpy.linalg.inv(J)), cU)[0, 0]

    if adj_and_org:
        return score_adj, score_org
    return score_adj
