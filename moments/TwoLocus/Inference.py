import numpy as np, numpy
import scipy, math
from scipy.special import gammaln
import scipy.optimize
from numpy import logical_and, logical_not
import os, sys
import moments.Numerics, moments.Misc

_counter = 0
_out_of_bounds_val = -1e12


def ll(model, data):
    """
    The log-likelihood of the data given the model linkage frequency spectrum
    """
    ll_arr = ll_per_bin(model, data)
    return ll_arr.sum()


def ll_per_bin(model, data):
    """
    Poisson log-likelihood of each entry in the data given the model sfs
    """
    result = -model.data + data.data * np.log(model) - gammaln(data + 1.0)
    return result


def ll_multinom(model, data):
    """
    LL of the data given the model, with optimal rescaling
    """
    ll_arr = ll_multinom_per_bin(model, data)
    return ll_arr.sum()


def ll_multinom_per_bin(model, data):
    """
    Multinomial log-likelihood of each entry in the data given the model
    """
    theta_opt = optimal_sfs_scaling(model, data)
    return ll_per_bin(theta_opt * model, data)


def optimal_sfs_scaling(model, data):
    """
    Optimal multiplicative scaling factor between model and data
    """
    model, data = moments.Numerics.intersect_masks(model, data)
    return data.sum() / model.sum()


def optimally_scaled_sfs(model, data):
    """
    Optimally scaled model to data
    """
    return optimal_sfs_scaling(model, data) * model


def ll_over_rho_bins(model_list, data_list):
    """
    The log-likelihood of the binned data given the model spectra for the same bins
    Input list of models for rho bins, and list of data for rho bins
    """
    if len(model_list) != len(data_list):
        print("model list and data list must be of same length")
        return 0
    LL = 0
    for ii in range(len(model_list)):
        LL += ll(model_list[ii], data_list[ii])
    return LL


def ll_over_rho_bins_multinom(model_list, data_list):
    """
    The log-likelihood of the binned data given the model spectra for the same bins
    Input list of models for rho bins, and list of data for rho bins
    """
    if len(model_list) != len(data_list):
        print("model list and data list must be of same length")
        return 0
    LL = 0
    for ii in range(len(model_list)):
        LL += ll_multinom(model_list[ii], data_list[ii])
    return LL


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


def _object_func(
    params,
    data_list,
    model_func,
    rhos=[0],
    lower_bound=None,
    upper_bound=None,
    verbose=0,
    multinom=True,
    flush_delay=0,
    func_args=[],
    func_kwargs={},
    fixed_params=None,
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

    ns = len(data_list[0]) - 1
    all_args = [params_up, ns]
    func_kwargs = func_kwargs.copy()
    func_kwargs["rhos"] = rhos
    model_list = model_func(*all_args, **func_kwargs)
    rho_mids = (np.array(rhos)[:-1] + np.array(rhos)[1:]) / 2
    func_kwargs["rhos"] = rho_mids
    model_list_mids = model_func(*all_args, **func_kwargs)

    # trap rule
    # model_list = [(m1+m2)/2. for m1,m2 in zip(model_list[:-1],model_list[1:])]
    # simp rule
    model_list = [
        (m1 + m2 + 4 * m3) / 6.0
        for m1, m2, m3 in zip(model_list[:-1], model_list[1:], model_list_mids)
    ]

    if multinom:
        result = ll_over_rho_bins_multinom(model_list, data_list)
    else:
        result = ll_over_rho_bins(model_list, data_list)

    # Bad result
    if numpy.isnan(result):
        result = _out_of_bounds_val

    if (verbose > 0) and (_counter % verbose == 0):
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params_up]))
        output_stream.write(
            "%-8i, %-12g, %s%s" % (_counter, result, param_str, os.linesep)
        )
        moments.Misc.delayed_flush(delay=flush_delay)

    return -result


def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(numpy.exp(log_params), *args, **kwargs)


def optimize_log_fmin(
    p0,
    data_list,
    model_func,
    rhos=[0],
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
    output_file=None,
):
    if output_file:
        output_stream = open(output_file, "w")
    else:
        output_stream = sys.stdout

    args = (
        data_list,
        model_func,
        rhos,
        lower_bound,
        upper_bound,
        verbose,
        multinom,
        flush_delay,
        func_args,
        func_kwargs,
        fixed_params,
        output_stream,
    )

    p0 = _project_params_down(p0, fixed_params)

    outputs = scipy.optimize.fmin(
        _object_func_log,
        numpy.log(p0),
        args=args,
        disp=False,
        maxiter=maxiter,
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
