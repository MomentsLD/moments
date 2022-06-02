"""
Comparison and optimization of model spectra to data.
"""
import logging

logger = logging.getLogger("Inference")

import os, sys

import numpy
from numpy import logical_and, logical_not

from . import Misc, Numerics
from scipy.special import gammaln
import scipy.optimize

#: Stores thetas
_theta_store = {}
#: Counts calls to object_func
_counter = 0
#: Returned when object_func is passed out-of-bounds params or gets a NaN ll.
_out_of_bounds_val = -1e8


def _object_func(
    params,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    multinom=True,
    flush_delay=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_stream=sys.stdout,
    store_thetas=False,
):
    """
    Objective function for optimization.
    """
    global _counter
    _counter += 1

    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)

    # Check our parameter bounds
    if lower_bound is not None:
        for pval, bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val / ll_scale
    if upper_bound is not None:
        for pval, bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val / ll_scale

    ns = data.sample_sizes
    all_args = [params_up, ns] + list(func_args)

    func_kwargs = func_kwargs.copy()
    sfs = model_func(*all_args, **func_kwargs)
    if multinom:
        result = ll_multinom(sfs, data)
    else:
        result = ll(sfs, data)

    if store_thetas:
        global _theta_store
        _theta_store[tuple(params)] = optimal_sfs_scaling(sfs, data)

    # Bad result
    if numpy.isnan(result):
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params_up]))
        output_stream.write(
            "%-8i, %-12g, %s%s" % (_counter, result, param_str, os.linesep)
        )
        Misc.delayed_flush(delay=flush_delay)

    return -result / ll_scale


def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(numpy.exp(log_params), *args, **kwargs)


def optimize_log(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    gtol=1e-5,
    multinom=True,
    maxiter=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using the BFGS method. This optimization
    method works well when we start reasonably close to the optimum.

    Because this works in log(params), it cannot explore values of params < 0.
    However, it should perform well when parameters range over different orders
    of magnitude.

    :param p0: Initial parameters.
    :param data: Data SFS.
    :param model_func: Function to evaluate model spectrum. Should take arguments
        ``model_func(params, (n1,n2...))``.
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0.
    :param verbose: If > 0, print optimization status every ``verbose`` steps.
    :param output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
    :param epsilon: Step-size to use for finite-difference derivatives.
    :param gtol: Convergence criterion for optimization. For more info,
        see help(scipy.optimize.fmin_bfgs)
    :param multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum iterations to run for.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
        Using func_args.
        For example, you could define your model function as
        ``def func((p1,p2), ns, f1, f2): ...``.
        If you wanted to fix f1=0.1 and f2=0.2 in the optimization, you
        would pass func_args = [0.1,0.2] (and ignore the fixed_params
        argument).
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        ll hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
        For example, suppose your model function is
        ``def func((p1,f1,p2,f2), ns): ...``
        If you wanted to fix f1=0.1 and f2=0.2 in the optimization,
        you would pass fixed_params = [None,0.1,None,0.2] (and ignore
        the func_args argument).
    :param ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
        too large. (This appears to be a flaw in the scipy
        implementation.) To overcome this, pass ll_scale > 1, which will
        simply reduce the magnitude of the log-likelihood. Once in a
        region of reasonable likelihood, you'll probably want to
        re-optimize with ll_scale=1.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(
        _object_func_log,
        numpy.log(p0),
        epsilon=epsilon,
        args=args,
        gtol=gtol,
        full_output=True,
        disp=False,
        maxiter=maxiter,
    )
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def optimize_log_lbfgsb(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    pgtol=1e-5,
    multinom=True,
    maxiter=1e5,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using the L-BFGS-B method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum. This method is
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

    :param p0: Initial parameters.
    :param data: Spectrum with data.
    :param model_function: Function to evaluate model spectrum. Should take arguments
        (params, (n1,n2...))
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param verbose: If > 0, print optimization status every <verbose> steps.
    :param output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
    :param epsilon: Step-size to use for finite-difference derivatives.
    :param pgtol: Convergence criterion for optimization. For more info,
        see help(scipy.optimize.fmin_l_bfgs_b)
    :param multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum algorithm iterations to run.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        will hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
    :param ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
        too large. (This appears to be a flaw in the scipy
        implementation.) To overcome this, pass ll_scale > 1, which will
        simply reduce the magnitude of the log-likelihood. Once in a
        region of reasonable likelihood, you'll probably want to
        re-optimize with ll_scale=1.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        None,
        None,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    # Make bounds list. For this method it needs to be in terms of log params.
    if lower_bound is None:
        lower_bound = [None] * len(p0)
    else:
        lower_bound = [numpy.log(lb) if lb is not None else None for lb in lower_bound]
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [None] * len(p0)
    else:
        upper_bound = [numpy.log(ub) if ub is not None else None for ub in upper_bound]
    upper_bound = _project_params_down(upper_bound, fixed_params)
    bounds = list(zip(lower_bound, upper_bound))

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin_l_bfgs_b(
        _object_func_log,
        numpy.log(p0),
        bounds=bounds,
        epsilon=epsilon,
        args=args,
        iprint=-1,
        pgtol=pgtol,
        maxiter=maxiter,
        approx_grad=True,
    )
    xopt, fopt, info_dict = outputs

    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, info_dict


def minus_ll(model, data):
    """
    The negative of the log-likelihood of the data given the model sfs.
    """
    return -ll(model, data)


def ll(model, data):
    """
    The log-likelihood of the data given the model sfs.

    Evaluate the log-likelihood of the data given the model. This is based on
    Poisson statistics, where the probability of observing k entries in a cell
    given that the mean number is given by the model is
    :math:`P(k) = exp(-model) * model^k / k!`.

    Note: If either the model or the data is a masked array, the return ll will
    ignore any elements that are masked in *either* the model or the data.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    ll_arr = ll_per_bin(model, data)
    return ll_arr.sum()


def ll_per_bin(model, data, missing_model_cutoff=1e-6):
    """
    The Poisson log-likelihood of each entry in the data given the model sfs.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    :param missing_model_cutoff: Due to numerical issues, there may be entries in the
        FS that cannot be stable calculated. If these entries
        involve a fraction of the data larger than
        missing_model_cutoff, a warning is printed.
    """
    if data.folded and not model.folded:
        model = model.fold()

    # Using numpy.ma.log here ensures that any negative or nan entries in model
    # yield masked entries in result. We can then check for correctness of
    # calculation by simply comparing masks.
    # Note: Using .data attributes directly saves a little computation time. We
    # use model and data as a whole at least once, to ensure masking is done
    # properly.
    result = -model.data + data.data * model.log() - gammaln(data + 1.0)
    if numpy.all(result.mask == data.mask):
        return result

    not_data_mask = logical_not(data.mask)
    data_sum = data.sum()

    missing = logical_and(model < 0, not_data_mask)
    if numpy.any(missing) and data[missing].sum() / data.sum() > missing_model_cutoff:
        logger.warn("Model is < 0 where data is not masked.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    # If the data is 0, it's okay for the model to be 0. In that case the ll
    # contribution is 0, which is fine.
    missing = logical_and(model == 0, logical_and(data > 0, not_data_mask))
    if numpy.any(missing) and data[missing].sum() / data_sum > missing_model_cutoff:
        logger.warn("Model is 0 where data is neither masked nor 0.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    missing = numpy.logical_and(model.mask, not_data_mask)
    if numpy.any(missing) and data[missing].sum() / data_sum > missing_model_cutoff:
        print(data[missing].sum(), data_sum)
        logger.warn("Model is masked in some entries where data is not.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    missing = numpy.logical_and(numpy.isnan(model), not_data_mask)
    if numpy.any(missing) and data[missing].sum() / data_sum > missing_model_cutoff:
        logger.warn("Model is nan in some entries where data is not masked.")
        logger.warn(
            "Number of affected entries is %i. Sum of data in those "
            "entries is %g:" % (missing.sum(), data[missing].sum())
        )

    return result


def ll_multinom_per_bin(model, data):
    """
    Mutlinomial log-likelihood of each entry in the data given the model.

    Scales the model sfs to have the optimal theta for comparison with the data.
    """
    theta_opt = optimal_sfs_scaling(model, data)
    return ll_per_bin(theta_opt * model, data)


def ll_multinom(model, data):
    """
    Log-likelihood of the data given the model, with optimal rescaling.

    Evaluate the log-likelihood of the data given the model. This is based on
    Poisson statistics, where the probability of observing k entries in a cell
    given that the mean number is given by the model is
    :math:`P(k) = exp(-model) * model^k / k!`.

    model is optimally scaled to maximize ll before calculation.

    Note: If either the model or the data is a masked array, the return ll will
    ignore any elements that are masked in *either* the model or the data.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    ll_arr = ll_multinom_per_bin(model, data)
    return ll_arr.sum()


def minus_ll_multinom(model, data):
    """
    The negative of the log-likelihood of the data given the model sfs.

    Return a double that is -(log-likelihood)
    """
    return -ll_multinom(model, data)


def linear_Poisson_residual(model, data, mask=None):
    """
    Return the Poisson residuals, (model - data)/sqrt(model), of model and data.

    mask sets the level in model below which the returned residual array is
    masked. The default of 0 excludes values where the residuals are not
    defined.

    In the limit that the mean of the Poisson distribution is large, these
    residuals are normally distributed. (If the mean is small, the Anscombe
    residuals are better.)

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    :param mask: Optional mask, with same size as ``model``.
    """
    if data.folded and not model.folded:
        model = model.fold()

    resid = (model - data) / numpy.ma.sqrt(model)
    if mask is not None:
        tomask = numpy.logical_and(model <= mask, data <= mask)
        resid = numpy.ma.masked_where(tomask, resid)
    return resid


def Anscombe_Poisson_residual(model, data, mask=None):
    """
    Return the Anscombe Poisson residuals between model and data.

    mask sets the level in model below which the returned residual array is
    masked. This excludes very small values where the residuals are not normal.
    1e-2 seems to be a good default for the NIEHS human data. (model = 1e-2,
    data = 0, yields a residual of ~1.5.)

    Residuals defined in this manner are more normally distributed than the
    linear residuals when the mean is small. See this reference below for
    justification: Pierce DA and Schafer DW, "Residuals in generalized linear
    models" Journal of the American Statistical Association, 81(396)977-986
    (1986).

    Note that I tried implementing the "adjusted deviance" residuals, but they
    always looked very biased for the cases where the data was 0.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    :param mask: Optional mask, with same size as ``model``.
    """
    if data.folded and not model.folded:
        model = model.fold()
    # Because my data have often been projected downward or averaged over many
    # iterations, it appears better to apply the same transformation to the data
    # and the model.
    # For some reason data**(-1./3) results in entries in data that are zero
    # becoming masked. Not just the result, but the data array itself. We use
    # the power call to get around that.
    # This seems to be a common problem, that we want to use numpy.ma functions
    # on masked arrays, because otherwise the mask on the input itself can be
    # changed. Subtle and annoying. If we need to create our own functions, we
    # can use numpy.ma.core._MaskedUnaryOperation.
    datatrans = data ** (2.0 / 3) - numpy.ma.power(data, -1.0 / 3) / 9
    modeltrans = model ** (2.0 / 3) - numpy.ma.power(model, -1.0 / 3) / 9
    resid = 1.5 * (datatrans - modeltrans) / model ** (1.0 / 6)
    if mask is not None:
        tomask = numpy.logical_and(model <= mask, data <= mask)
        tomask = numpy.logical_or(tomask, data == 0)
        resid = numpy.ma.masked_where(tomask, resid)
    # It makes more sense to me to have a minus sign here... So when the
    # model is high, the residual is positive. This is opposite of the
    # Pierce and Schafner convention.
    return -resid


def optimally_scaled_sfs(model, data):
    """
    Optimially scale model sfs to data sfs.

    Returns a new scaled model sfs.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    return optimal_sfs_scaling(model, data) * model


def optimal_sfs_scaling(model, data):
    """
    Optimal multiplicative scaling factor between model and data.

    This scaling is based on only those entries that are masked in neither
    model nor data.

    :param model: The model Spectrum object.
    :param data: The data Spectrum object, with same size as ``model``.
    """
    if data.folded and not model.folded:
        model = model.fold()

    model, data = Numerics.intersect_masks(model, data)
    return data.sum() / model.sum()


def optimize_log_fmin(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    multinom=True,
    maxiter=None,
    maxfun=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using Nelder-Mead.
    This optimization method may work better than BFGS when far from a
    minimum. It is much slower, but more robust, because it doesn't use
    gradient information.

    Because this works in log(params), it cannot explore values of params < 0.
    It should also perform better when parameters range over large scales.

    :param p0: Initial parameters.
    :param data: Spectrum with data.
    :param model_function: Function to evaluate model spectrum. Should take arguments
        (params, (n1,n2...))
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param verbose: If True, print optimization status every <verbose> steps.
    :param output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
    :param multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum number of iterations to run optimization.
    :param maxfun: Maximum number of objective function calls to perform.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        will hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        1.0,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin(
        _object_func_log,
        numpy.log(p0),
        args=args,
        disp=False,
        maxiter=maxiter,
        maxfun=maxfun,
        full_output=True,
    )
    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, iter, funcalls, warnflag


def optimize_powell(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    xtol=1e-4,
    ftol=1e-4,
    multinom=True,
    maxiter=None,
    maxfunc=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
    retall=False,
):
    """
    Optimize parameters using Powell's conjugate direction method.

    This method works without calculating any derivatives, and optimizes along
    one direction at a time. May be useful as an initial search for an approximate
    solution, followed by further optimization using a gradient optimizer.

    p0: Initial parameters.
    data: Spectrum with data.
    model_func: Function to evaluate model spectrum. Should take arguments
                (params, (n1,n2...)).
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0.
    verbose: If > 0, print optimization status every <verbose> steps.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    xtol: Error tolerance for line search.
    ftol: Relative error acceptable for convergence.
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum iterations to run for.
    maxfunc: Maximum number of function evalutions.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_powell).
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, and its second argument is an array of sample sizes
               for the sfs.
               For example, you could define your model function as
               def func((p1,p2), ns, f1, f2):
                   ....
               If you wanted to fix f1=0.1 and f2=0.2 in the optimization, you
               would pass func_args = [0.1,0.2].
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
                  For example, suppose your model function is
                  def func((p1,f1,p2,f2), ns):
                      ....
                  If you wanted to fix f1=0.1 and f2=0.2 in the optimization,
                  you would pass fixed_params = [None,0.1,None,0.2].
    ll_scale: The algorithm may fail if your initial log-likelihood is
              too large. To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    retall: If True, return a list of solutions at each iteration.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_powell(
        _object_func,
        p0,
        args=args,
        xtol=xtol,
        ftol=ftol,
        maxiter=maxiter,
        maxfun=maxfunc,
        disp=False,
        full_output=True,
        retall=retall,
    )
    if retall:
        xopt, fopt, direc, iters, funcalls, warnflag, allvecs = outputs
    else:
        xopt, fopt, direc, iters, funcalls, warnflag = outputs
    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    elif retall:
        return xopt, fopt, direc, iters, funcalls, warnflag, allvecs
    else:
        return xopt, fopt, direc, iters, funcalls, warnflag


def optimize_log_powell(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    multinom=True,
    maxiter=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    output_file=None,
):
    """
    Optimize log(params) to fit model to data using Powell's conjugate direction method.

    This method works without calculating any derivatives, and optimizes along
    one direction at a time. May be useful as an initial search for an approximate
    solution, followed by further optimization using a gradient optimizer.

    Because this works in log(params), it cannot explore values of params < 0.

    :param p0: Initial parameters.
    :param data: Spectrum with data.
    :param model_function: Function to evaluate model spectrum. Should take arguments
        (params, (n1,n2...))
    :param lower_bound: Lower bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param upper_bound: Upper bound on parameter values. If not None, must be of same
        length as p0. A parameter can be declared unbound by assigning
        a bound of None.
    :param verbose: If True, print optimization status every <verbose> steps.
        output_file: Stream verbose output into this filename. If None, stream to
        standard out.
    :param flush_delay: Standard output will be flushed once every <flush_delay>
        minutes. This is useful to avoid overloading I/O on clusters.
        multinom: If True, do a multinomial fit where model is optimially scaled to
        data at each step. If False, assume theta is a parameter and do
        no scaling.
    :param maxiter: Maximum iterations to run for.
    :param full_output: If True, return full outputs as in described in
        help(scipy.optimize.fmin_bfgs)
    :param func_args: Additional arguments to model_func. It is assumed that
        model_func's first argument is an array of parameters to
        optimize, that its second argument is an array of sample sizes
        for the sfs, and that its last argument is the list of grid
        points to use in evaluation.
    :param func_kwargs: Additional keyword arguments to model_func.
    :param fixed_params: If not None, should be a list used to fix model parameters at
        particular values. For example, if the model parameters
        are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
        will hold nu1=0.5 and m=2. The optimizer will only change
        T and m. Note that the bounds lists must include all
        parameters. Optimization will fail if the fixed values
        lie outside their bounds. A full-length p0 should be passed
        in; values corresponding to fixed parameters are ignored.
        (See help(moments.Inference.optimize_log for examples of func_args and
        fixed_params usage.)
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        1.0,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_powell(
        _object_func_log,
        numpy.log(p0),
        args=args,
        disp=False,
        maxiter=maxiter,
        full_output=True,
    )
    xopt, fopt, direc, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(numpy.exp(xopt), fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, direc, iter, funcalls, warnflag


def optimize(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    gtol=1e-5,
    multinom=True,
    maxiter=None,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize params to fit model to data using the BFGS method.

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, (n1,n2...))
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    gtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_bfgs)
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum iterations to run for.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(moments.Inference.optimize_log for examples of func_args and
     fixed_params usage.)
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_bfgs(
        _object_func,
        p0,
        epsilon=epsilon,
        args=args,
        gtol=gtol,
        full_output=True,
        disp=False,
        maxiter=maxiter,
    )
    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = outputs
    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag


def optimize_lbfgsb(
    p0,
    data,
    model_func,
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    flush_delay=0.5,
    epsilon=1e-3,
    pgtol=1e-5,
    multinom=True,
    maxiter=1e5,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    ll_scale=1,
    output_file=None,
):
    """
    Optimize params to fit model to data using the L-BFGS-B method.

    Note: this optimization method can explore negative values. You must therefore
    specify lower bounds for values that cannot take negative numbers (such
    as event times, population sizes, and migration rates).

    This optimization method works well when we start reasonably close to the
    optimum. It is best at burrowing down a single minimum. This method is
    better than optimize_log if the optimum lies at one or more of the
    parameter bounds. However, if your optimum is not on the bounds, this
    method may be much slower.

    p0: Initial parameters.
    data: Spectrum with data.
    model_function: Function to evaluate model spectrum. Should take arguments
                    (params, (n1,n2...))
    lower_bound: Lower bound on parameter values. If not None, must be of same
                 length as p0. A parameter can be declared unbound by assigning
                 a bound of None.
    upper_bound: Upper bound on parameter values. If not None, must be of same
                 length as p0. A parameter can be declared unbound by assigning
                 a bound of None.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    epsilon: Step-size to use for finite-difference derivatives.
    pgtol: Convergence criterion for optimization. For more info,
          see help(scipy.optimize.fmin_l_bfgs_b)
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    maxiter: Maximum algorithm iterations evaluations to run.
    full_output: If True, return full outputs as in described in
                 help(scipy.optimize.fmin_bfgs)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(moments.Inference.optimize_log for examples of func_args and
     fixed_params usage.)
    ll_scale: The bfgs algorithm may fail if your initial log-likelihood is
              too large. (This appears to be a flaw in the scipy
              implementation.) To overcome this, pass ll_scale > 1, which will
              simply reduce the magnitude of the log-likelihood. Once in a
              region of reasonable likelihood, you'll probably want to
              re-optimize with ll_scale=1.

    The L-BFGS-B method was developed by Ciyou Zhu, Richard Byrd, and Jorge
    Nocedal. The algorithm is described in:
      * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
        Constrained Optimization, (1995), SIAM Journal on Scientific and
        Statistical Computing , 16, 5, pp. 1190-1208.
      * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
        FORTRAN routines for large scale bound constrained optimization (1997),
        ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550-560.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        None,
        None,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        ll_scale,
        output_stream,
    )

    # Make bounds list. For this method it needs to be in terms of log params.
    if lower_bound is None:
        lower_bound = [None] * len(p0)
    lower_bound = _project_params_down(lower_bound, fixed_params)
    if upper_bound is None:
        upper_bound = [None] * len(p0)
    upper_bound = _project_params_down(upper_bound, fixed_params)
    bounds = list(zip(lower_bound, upper_bound))

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin_l_bfgs_b(
        _object_func,
        p0,
        bounds=bounds,
        epsilon=epsilon,
        args=args,
        iprint=-1,
        pgtol=pgtol,
        maxiter=maxiter,
        approx_grad=True,
    )
    xopt, fopt, info_dict = outputs

    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, info_dict


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

    return numpy.array(pout)


def _project_params_up(pin, fixed_params):
    """
    Fold fixed parameters into pin.
    """
    if fixed_params is None:
        return pin

    if numpy.isscalar(pin):
        pin = [pin]

    pout = numpy.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout


index_exp = numpy.index_exp


def optimize_grid(
    data,
    model_func,
    grid,
    verbose=0,
    flush_delay=0.5,
    multinom=True,
    full_output=False,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
    output_file=None,
):
    """
    Optimize params to fit model to data using brute force search over a grid.

    data: Spectrum with data.
    model_func: Function to evaluate model spectrum. Should take arguments
                (params, (n1,n2...))
    grid: Grid of parameter values over which to evaluate likelihood. See
          below for specification instructions.
    verbose: If > 0, print optimization status every <verbose> steps.
    output_file: Stream verbose output into this filename. If None, stream to
                 standard out.
    flush_delay: Standard output will be flushed once every <flush_delay>
                 minutes. This is useful to avoid overloading I/O on clusters.
    multinom: If True, do a multinomial fit where model is optimially scaled to
              data at each step. If False, assume theta is a parameter and do
              no scaling.
    full_output: If True, return popt, llopt, grid, llout, thetas. Here popt is
                 the best parameter set found and llopt is the corresponding
                 (composite) log-likelihood. grid is the array of parameter
                 values tried, llout is the corresponding log-likelihoods, and
                 thetas is the corresponding thetas. Note that the grid includes
                 only the parameters optimized over, and that the order of
                 indices is such that grid[:,0,2] would be a set of parameters
                 if two parameters were optimized over. (Note the : in the
                 first index.)
    func_args: Additional arguments to model_func. It is assumed that
               model_func's first argument is an array of parameters to
               optimize, that its second argument is an array of sample sizes
               for the sfs, and that its last argument is the list of grid
               points to use in evaluation.
    func_kwargs: Additional keyword arguments to model_func.
    fixed_params: If not None, should be a list used to fix model parameters at
                  particular values. For example, if the model parameters
                  are (nu1,nu2,T,m), then fixed_params = [0.5,None,None,2]
                  will hold nu1=0.5 and m=2. The optimizer will only change
                  T and m. Note that the bounds lists must include all
                  parameters. Optimization will fail if the fixed values
                  lie outside their bounds. A full-length p0 should be passed
                  in; values corresponding to fixed parameters are ignored.
    (See help(moments.Inference.optimize_log for examples of func_args and
     fixed_params usage.)

    Search grids are specified using a moments.Inference.index_exp object (which
    is an alias for numpy.index_exp). The grid is specified by passing a range
    of values for each parameter. For example, index_exp[0:1.1:0.3,
    0.7:0.9:11j] will search over parameter 1 with values 0,0.3,0.6,0.9 and
    over parameter 2 with 11 points between 0.7 and 0.9 (inclusive). (Notice
    the 11j in the second parameter range specification.) Note that the grid
    list should include only parameters that are optimized over, not fixed
    parameter values.
    """
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data,
        model_func,
        None,
        None,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        1.0,
        output_stream,
        full_output,
    )

    if full_output:
        global _theta_store
        _theta_store = {}

    outputs = scipy.optimize.brute(
        _object_func, ranges=grid, args=args, full_output=full_output, finish=False
    )
    if full_output:
        xopt, fopt, grid, fout = outputs
        # Thetas are stored as a dictionary, because we can't guarantee
        # iteration order in brute(). So we have to iterate back over them
        # to produce the proper order to return.
        thetas = numpy.zeros(fout.shape)
        for indices, temp in numpy.ndenumerate(fout):
            # This is awkward, because we need to access grid[:,indices]
            grid_indices = tuple([slice(None, None, None)] + list(indices))
            thetas[indices] = _theta_store[tuple(grid[grid_indices])]
    else:
        xopt = outputs
    xopt = _project_params_up(xopt, fixed_params)

    if output_file:
        output_stream.close()

    if not full_output:
        return xopt
    else:
        return xopt, fopt, grid, fout, thetas


def add_misid_param(func):
    def misid_func(params, *args, **kwargs):
        misid = params[-1]
        fs = func(params[:-1], *args, **kwargs)
        return (1 - misid) * fs + misid * Numerics.reverse_array(fs)

    return misid_func
