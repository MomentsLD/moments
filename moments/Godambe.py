"""
Parameter uncertainties and likelihood ratio tests using Godambe information.
"""

import warnings

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
    Calculate the Hessian matrix of partial second derivatives of ``func``, where
    entry ``(i, j)`` is the second derivative of ``func`` with respect to parameters
    ``i`` and ``j``, computed by finite differences.

    :param func: Function returning a scalar, called as ``func(params, *args)``.
    :type func: function
    :param p0: Parameter values to take derivatives around.
    :type p0: list-like
    :param eps: Fractional stepsize to use when taking finite-difference
        derivatives. Parameters equal to zero use a step size of ``eps``.
    :type eps: float
    :param args: Additional arguments to ``func``.
    :type args: tuple
    :return: The Hessian matrix, with shape ``(len(p0), len(p0))``.
    :rtype: numpy.ndarray
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


# Threshold on the smallest eigenvalue of the correlation-scaled Hessian of the
# complex model at the nested fit, below which LRT_adjust warns. This tolerance was
# calibrated on a two-population split-with-migration model.
_HESSIAN_WARN_TOL = 0.005


def _check_nested_hessian(H, labels, caller):
    """
    Warn if the Hessian of the complex model at the best fit of the nested model is
    indefinite or nearly singular, in which case the LRT adjustment is unreliable.

    :param H: The Hessian of the negative log-likelihood of the complex model, over
        all of its parameters, at the best fit of the nested model.
    :param labels: A name for each parameter, used in the warning.
    :param caller: The name of the calling function, used in the warning.
    :return: True if a warning was issued.
    """
    H = numpy.asarray(H, dtype=float)
    labels = list(labels)
    advice = (
        "This often happens when the nested model fits the data poorly, so that "
        "the quadratic approximation that the adjustment relies on does not hold. "
        "The adjustment factor may be unreliable. See the documentation on "
        "likelihood ratio tests for alternatives."
    )
    if not numpy.all(numpy.isfinite(H)):
        warnings.warn(
            f"{caller}: the Hessian of the complex model at the nested fit has "
            f"non-finite entries. {advice}",
            stacklevel=3,
        )
        return True
    d = numpy.diag(H)
    if numpy.any(d <= 0):
        bad = [labels[i] for i in numpy.where(d <= 0)[0]]
        warnings.warn(
            f"{caller}: the log-likelihood of the complex model at the nested fit "
            f"is not concave in {', '.join(bad)} (non-positive diagonal entries "
            f"of the Hessian). {advice}",
            stacklevel=3,
        )
        return True
    s = numpy.sqrt(d)
    w, v = numpy.linalg.eigh(H / numpy.outer(s, s))
    if w[0] < _HESSIAN_WARN_TOL:
        # the parameters that make up most of the near-null direction
        cut = min(0.3, numpy.abs(v[:, 0]).max())
        direction = [l for l, x in zip(labels, v[:, 0]) if abs(x) >= cut]
        kind = "indefinite" if w[0] <= 0 else "nearly singular"
        warnings.warn(
            f"{caller}: the Hessian of the complex model at the nested fit is "
            f"{kind} (smallest eigenvalue of the correlation-scaled Hessian "
            f"{w[0]:.2g}), along a direction mostly in {', '.join(direction)}. "
            f"{advice}",
            stacklevel=3,
        )
        return True
    return False


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
    :type return_GIM: bool
    :return: The standard deviations of the parameters (including theta, last, if
        ``multinom`` is True), and if ``return_GIM`` is True, the GIM.
    :rtype: numpy.ndarray, or tuple of numpy.ndarray
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
    :return: The standard deviations of the parameters (including theta, last, if
        ``multinom`` is True).
    :rtype: numpy.ndarray
    """
    if multinom:
        func_multi = func_ex
        model = func_multi(p0, data.sample_sizes)
        theta_opt = Inference.optimal_sfs_scaling(model, data)
        p0 = list(p0) + [theta_opt]
        func_ex = lambda p, ns: p[-1] * func_multi(p[:-1], ns)
    H = _get_godambe(func_ex, [], p0, data, eps, log, just_hess=True)
    return numpy.sqrt(numpy.diag(numpy.linalg.inv(H)))


def LRT_adjust(
    func_ex,
    all_boot,
    p0,
    data,
    nested_indices,
    multinom=True,
    eps=0.01,
    check_hessian=True,
):
    """
    First-order moment matching adjustment factor for likelihood ratio test.

    The adjusted likelihood ratio test statistic is
    ``D_adj = adj * 2 * (ll_complex - ll_simple)``, where ``ll_complex`` and
    ``ll_simple`` are the log-likelihoods of the best fits of the complex and simple
    (nested) models (Coffman et al., 2016). Its p-value is given by
    ``sum_chi2_ppf(D_adj, weights)``.

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
    :param check_hessian: If True, also compute the Hessian of the complex model over
        all of its parameters at ``p0``, and warn if it is indefinite or nearly
        singular, in which case the adjustment is unreliable (see the documentation
        on likelihood ratio tests). This takes roughly ``2 * k**2`` extra model
        evaluations for ``k`` parameters. The adjustment factor is the same either
        way.
    :type check_hessian: bool
    :return: The adjustment factor, ``adj``.
    :rtype: float
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

    labels = [f"p0[{i}]" for i in range(len(p0))]
    if multinom:
        labels[-1] = "theta"
    if check_hessian:
        H_full = _get_godambe(func_ex, [], p0, data, eps, log=False, just_hess=True)
        _check_nested_hessian(H_full, labels, "moments.Godambe.LRT_adjust")
    else:
        _check_nested_hessian(
            H, [labels[i] for i in nested_indices], "moments.Godambe.LRT_adjust"
        )

    adjust = len(nested_indices) / numpy.trace(numpy.dot(J, numpy.linalg.inv(H)))
    return adjust


def sum_chi2_ppf(x, weights=(0, 1)):
    """
    Returns the upper tail probability (1 - cdf) of a weighted sum of chi^2
    distributions, which is the p-value of a likelihood ratio test statistic.
    (Despite the name, this is not the percent point function.)

    :param x: Value(s) at which to evaluate the upper tail probability, e.g. the
        (adjusted) test statistic.
    :type x: float or array-like
    :param weights: Weights of the chi^2 distributions, beginning with zero degrees
        of freedom. For example, ``weights=(0, 1)`` is the chi^2 distribution with
        one degree of freedom. For single parameters on the boundary, the correct
        distribution for the LRT is 0.5*chi^2_0 + 0.5*chi^2_1, which would be
        ``weights=(0.5, 0.5)``. The weights must sum to one.
    :type weights: list-like
    :return: The upper tail probability (p-value), with the same shape as ``x``.
    :rtype: float or numpy.ndarray
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
    Calculate the test statistic of the Wald test, adjusted using the Godambe
    Information Matrix.

    :param func_ex: Model function for the complex model.
    :type func_ex: demographic model
    :param all_boot: List of bootstrap frequency spectra.
    :type all_boot: list of spectra
    :param p0: Best-fit parameters for the simple model, with the nested parameters
        explicitly defined. Although equal to the values for the simple model, they
        should be given as a list that can be taken in by the complex model.
    :type p0: list-like
    :param data: Original data frequency spectrum.
    :type data: spectrum object
    :param nested_indices: List of positions of the nested parameters in the
        complex model's parameter list.
    :type nested_indices: list of ints
    :param full_params: Values of the parameters found only in the complex model.
        This can either be an array with just the values found only in the complex
        model, or the entire list of parameters of the complex model.
    :type full_params: list-like
    :param multinom: If True, assume model is defined without an explicit parameter
        for theta. Because uncertainty in theta must be accounted for to get
        correct uncertainties for other parameters, this function will
        automatically consider theta if multinom=True.
    :type multinom: bool
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
    :type eps: float
    :param adj_and_org: If False, return only the adjusted Wald statistic. If True,
        also return the unadjusted statistic as a second return value.
    :type adj_and_org: bool
    :return: The adjusted Wald statistic, and if ``adj_and_org`` is True, the
        unadjusted statistic.
    :rtype: float, or tuple of floats
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
    Calculate the test statistic of the score test, adjusted using the Godambe
    Information Matrix.

    :param func_ex: Model function for the complex model.
    :type func_ex: demographic model
    :param all_boot: List of bootstrap frequency spectra.
    :type all_boot: list of spectra
    :param p0: Best-fit parameters for the simple model, with the nested parameters
        explicitly defined. Although equal to the values for the simple model, they
        should be given as a list that can be taken in by the complex model.
    :type p0: list-like
    :param data: Original data frequency spectrum.
    :type data: spectrum object
    :param nested_indices: List of positions of the nested parameters in the
        complex model's parameter list.
    :type nested_indices: list of ints
    :param multinom: If True, assume model is defined without an explicit parameter
        for theta. Because uncertainty in theta must be accounted for to get
        correct uncertainties for other parameters, this function will
        automatically consider theta if multinom=True.
    :type multinom: bool
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
    :type eps: float
    :param adj_and_org: If False, return only the adjusted score statistic. If
        True, also return the unadjusted statistic as a second return value.
    :type adj_and_org: bool
    :return: The adjusted score statistic, and if ``adj_and_org`` is True, the
        unadjusted statistic.
    :rtype: float, or tuple of floats
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
