"""
Parameter uncertainties and likelihood ratio tests using Godambe information.
"""

import warnings

import numpy

from . import Inference
from .Spectrum_mod import Spectrum


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


# Finite-difference stencils: offsets (in steps h) and weights, so that a first
# derivative is sum(w * f(p + o * h)) / h, and a second derivative
# sum(w * f(p + o * h)) / h ** 2. "forward" is first order (the default for
# one-sided derivatives), and "forward2" second order; with a negative h, they are
# downward differences.
_FIRST = {
    "central": ((1, -1), (0.5, -0.5)),
    "forward": ((1, 0), (1, -1)),
    "forward2": ((0, 1, 2), (-1.5, 2, -0.5)),
}
_SECOND = {
    "central": ((1, 0, -1), (1, -2, 1)),
    "forward": ((2, 1, 0), (1, -2, 1)),
    "forward2": ((0, 1, 2, 3), (2, -5, 4, -1)),
}


def _stencils(p0, eps, log=False, side=None, scale=None):
    """
    The (signed) step size and the stencil of each parameter (see ``get_hess``).
    """
    steps, one_sided = _step_sizes(p0, eps, log=log)
    if scale is not None:
        steps = numpy.where(one_sided, eps * numpy.abs(scale), steps)
    if side is None:
        return steps, ["forward" if o else "central" for o in one_sided]
    steps = numpy.where(numpy.asarray(side) < 0, -steps, steps)
    return steps, ["central" if s == 0 else "forward2" for s in side]


def _evaluate(func, p0, shifts, args=(), f0=None):
    # func at p0, with parameters i shifted by shifts[i]
    if f0 is not None and not any(shifts.values()):
        return f0
    pwork = numpy.array(p0, copy=True, dtype=float)
    for i, shift in shifts.items():
        pwork[i] = p0[i] + shift
    return func(pwork, *args)


def _along(func, p0, i, steps, stencil, args=(), f0=None):
    # the weighted sum of func at offsets of parameter i, for a stencil
    offsets, weights = stencil
    return sum(
        w * _evaluate(func, p0, {i: o * steps[i]}, args, f0)
        for o, w in zip(offsets, weights)
    )


def _hessian_elem(func, f0, p0, ii, jj, steps, kinds, args=()):
    """
    Element [ii][jj] of the Hessian matrix of ``func`` at p0 (``f0``), from the
    parameters' step sizes and stencils (``_stencils``). Off the diagonal, the
    stencil is the product of the two first-derivative stencils.
    """
    if ii == jj:
        return (
            _along(func, p0, ii, steps, _SECOND[kinds[ii]], args, f0) / steps[ii] ** 2
        )
    ki, kj = kinds[ii], kinds[jj]
    if "forward" in (ki, kj):
        # first-order one-sided derivatives step both parameters forward
        ki = kj = "forward"
    (oi, wi), (oj, wj) = _FIRST[ki], _FIRST[kj]
    total = sum(
        a * b * _evaluate(func, p0, {ii: x * steps[ii], jj: y * steps[jj]}, args, f0)
        for x, a in zip(oi, wi)
        for y, b in zip(oj, wj)
    )
    return total / (steps[ii] * steps[jj])


def get_hess(func, p0, eps, args=(), log=False, side=None, scale=None):
    """
    Calculate the Hessian matrix of partial second derivatives of ``func``, where
    entry ``(i, j)`` is the second derivative of ``func`` with respect to parameters
    ``i`` and ``j``, computed by finite differences.

    :param func: Function returning a scalar, called as ``func(params, *args)``.
    :type func: function
    :param p0: Parameter values to take derivatives around.
    :type p0: list-like
    :param eps: Fractional stepsize to use when taking finite-difference
        derivatives (see ``_step_sizes``). Parameters equal to zero use a step size
        of ``eps``.
    :type eps: float
    :param args: Additional arguments to ``func``.
    :type args: tuple
    :param log: If True, ``p0`` are log-parameters, and the step size is ``eps``.
    :type log: bool
    :param side: Optional, one entry per parameter: 0 for central differences, 1
        for second-order one-sided differences upwards (for a parameter on a lower
        bound), -1 downwards (on an upper bound). By default, parameters equal to
        zero use first-order forward differences, and others central differences.
    :type side: list-like
    :param scale: Optional, one entry per parameter: the step size of parameters
        equal to zero is ``eps`` times their ``scale`` (by default, ``eps``).
    :type scale: list-like
    :return: The Hessian matrix, with shape ``(len(p0), len(p0))``.
    :rtype: numpy.ndarray
    """
    steps, kinds = _stencils(p0, eps, log=log, side=side, scale=scale)
    f0 = func(p0, *args)
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = _hessian_elem(func, f0, p0, ii, jj, steps, kinds, args)
            hess[jj][ii] = hess[ii][jj]
    return hess


def _get_grad(func, p0, eps, args=(), log=False, side=None, scale=None):
    """
    Calculate gradient vector

    :param func: Model function
    :param p0: Parameters for func
    :param eps: Fractional stepsize to use when taking finite-difference derivatives
        (see ``_step_sizes``).
    :param args: Additional arguments to func
    :param log: If True, ``p0`` are log-parameters, and the step size is ``eps``.
    :param side, scale: Optional, as in ``get_hess``.
    """
    steps, kinds = _stencils(p0, eps, log=log, side=side, scale=scale)
    grad = numpy.empty([len(p0), 1])
    for ii in range(len(p0)):
        grad[ii] = _along(func, p0, ii, steps, _FIRST[kinds[ii]], args) / steps[ii]
    return grad


def _expected_number_of_calls(params):
    n = len(params)
    return 4 * n * (n - 1) // 2 + 2 * n + 1


def _godambe_matrices(
    func,
    p0,
    data_args,
    all_boot,
    boot_args,
    eps,
    log=False,
    just_hess=False,
    side=None,
    scale=None,
):
    """
    Godambe information, Hessian, J and mean score (cU) matrices, for the
    log-likelihood ``func(params, *args)``: the Hessian with ``data_args``, and the
    scores with ``boot_args(boot)`` for each bootstrap in ``all_boot``. Used by the
    ``_get_godambe`` functions here and in moments.LD.Godambe and moments.Demes.

    :param log: If True, calculate derivatives in terms of log-parameters.
    :param just_hess: If True, only evaluate and return the Hessian matrix.
    :param side, scale: Optional, as in ``get_hess``.
    """
    if log:

        def f(logparams, *args):
            return func(numpy.exp(logparams), *args)

        p = numpy.log(p0)
    else:
        f, p = func, p0
    kwargs = dict(log=log, side=side, scale=scale)

    # First calculate the observed hessian
    hess = -get_hess(f, p, eps, args=data_args, **kwargs)
    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = numpy.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = numpy.zeros((len(p0), 1))
    for boot in all_boot:
        grad_temp = _get_grad(f, p, eps, args=boot_args(boot), **kwargs)
        J = J + numpy.outer(grad_temp, grad_temp)
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = numpy.linalg.inv(J)
    godambe = numpy.dot(numpy.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


def _get_godambe(
    func_ex, all_boot, p0, data, eps, log=False, just_hess=False, side=None, scale=None
):
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
    side, scale: Optional, as in ``get_hess``
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

    return _godambe_matrices(
        func,
        p0,
        [data],
        all_boot,
        lambda boot: [Spectrum(boot)],
        eps,
        log,
        just_hess,
        side=side,
        scale=scale,
    )


def _check_nested_hessian(H, labels, caller):
    """
    Warn if the Hessian of the complex model at the best fit of the nested model is
    indefinite (or has non-positive or non-finite entries on its diagonal), in which
    case the LRT adjustment is unreliable. A Hessian that is positive definite but
    nearly singular, as with strongly confounded parameters, does not warn: in
    simulations, the adjustment could still be well calibrated in that case.

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
    if w[0] <= 0:
        # the parameters that make up most of the direction of negative curvature
        cut = min(0.3, numpy.abs(v[:, 0]).max())
        direction = [l for l, x in zip(labels, v[:, 0]) if abs(x) >= cut]
        warnings.warn(
            f"{caller}: the Hessian of the complex model at the nested fit is "
            f"indefinite (smallest eigenvalue of the correlation-scaled Hessian "
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
        all of its parameters at ``p0``, and warn if it is indefinite, in which case
        the adjustment is unreliable (see the documentation on likelihood ratio
        tests). This takes roughly ``2 * k**2`` extra model
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
