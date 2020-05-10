"""
Parameter uncertainties are computed using Godambe information, described in
Coffman et al, MBE (2016). doi: 

If you use moments.LD.Godambe to compute parameter uncertainties, please cite
that paper. This was first developed by Alec Coffman for computing uncertainties
from inferences performed with dadi, modified here to handle LD decay curves.
"""

import numpy as np, numpy
from moments.LD import Inference
from moments.LD.LDstats_mod import LDstats

def hessian_elem(func, f0, p0, ii, jj, eps, args=(), one_sided=None):
    """
    Calculate element [ii][jj] of the Hessian matrix, a matrix
    of partial second derivatives w.r.t. to parameters ii and jj
        
    func: Model function
    f0: Evaluation of func at p0
    p0: Parameters for func
    eps: List of absolute step sizes to use for each parameter when taking
         finite differences.
    args: Additional arguments to func
    one_sided: Optionally, pass in a sequence of length p0 that determines
               whether a one-sided derivative will be used for each parameter.
    """
    # Note that we need to specify dtype=float, to avoid this being an integer
    # array which will silently fail when adding fractional eps.
    if one_sided is None:
        one_sided = [False]*len(p0)

    pwork = numpy.array(p0, copy=True, dtype=float)
    if ii == jj:
        if pwork[ii] != 0 and not one_sided[ii]:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)
            
            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork, *args)
            
            element = (fp - 2*f0 + fm)/eps[ii]**2
        else:
            pwork[ii] = p0[ii] + 2*eps[ii]
            fpp = func(pwork, *args)
            
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            element = (fpp - 2*fp + f0)/eps[ii]**2
    else:
        if pwork[ii] != 0 and pwork[jj] != 0 and not one_sided[ii] and not one_sided[jj]:
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

            element = (fpp - fpm - fmp + fmm)/(4 * eps[ii]*eps[jj])
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
            
            element = (fpp - fpm - fmp + f0)/(eps[ii]*eps[jj])
    return element

def get_hess(func, p0, eps, args=()):
    """
    Calculate Hessian matrix of partial second derivatives. 
    Hij = dfunc/(dp_i dp_j)
    
    func: Model function
    p0: Parameter values to take derivative around
    eps: Fractional stepsize to use when taking finite-difference derivatives
         Note that if eps*param is < 1e-6, then the step size for that parameter
         will simply be eps, to avoid numerical issues with small parameter
         perturbations.
    args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    one_sided = [False]*len(p0)
    for i, pval in enumerate(p0):
        if pval != 0:
            # Account for floating point arithmetic issues
            if pval*eps_in < 1e-6:
                eps[i] = eps_in
                one_sided[i] = True
            else:
                eps[i] = eps_in*pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    f0 = func(p0, *args)
    hess = numpy.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = hessian_elem(func, f0, p0, ii, jj, eps, args=args, one_sided=one_sided)
            hess[jj][ii] = hess[ii][jj]
    return hess

def get_grad(func, p0, eps, args=()):
    """
    Calculate gradient vector
    
    func: Model function
    p0: Parameters for func
    eps: Fractional stepsize to use when taking finite-difference derivatives
         Note that if eps*param is < 1e-6, then the step size for that parameter
         will simply be eps, to avoid numerical issues with small parameter
         perturbations.
    args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = numpy.empty([len(p0)])
    one_sided = [False]*len(p0)
    for i, pval in enumerate(p0):
        if pval != 0:
            # Account for floating point arithmetic issues
            if pval*eps_in < 1e-6:
                eps[i] = eps_in
                one_sided[i] = True
            else:
                eps[i] = eps_in*pval
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

            grad[ii] = (fp - fm)/(2*eps[ii])
        else:
            # Do one-sided finite-difference 
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            pwork[ii] = p0[ii]
            fm = func(pwork, *args)

            grad[ii] = (fp - fm)/eps[ii]
    return grad

ld_cache = {}
def get_godambe(func_ex, all_boot, p0, ms, vcs, eps, statistics, log=False,
                just_hess=False):
    """
    Godambe information and Hessian matrices

    func_ex: Model function
    all_boot: List of bootstrap frequency spectra
    p0: Best-fit parameters for func_ex.
    ms, vcs: Original data
    eps: Fractional stepsize to use when taking finite-difference derivatives
         Note that if eps*param is < 1e-6, then the step size for that parameter
         will simply be eps, to avoid numerical issues with small parameter
         perturbations.
    log: If True, calculate derivatives in terms of log-parameters
    just_hess: If True, only evaluate and return the Hessian matrix
    """
    # Cache evaluations of the LDstats inside our hessian/J 
    # evaluation function
    def func(params, m, v):
        key = tuple(params)
        if key not in ld_cache:
            ld_cache[key] = func_ex(params, statistics)
        y = ld_cache[key]
        return Inference.ll_over_bins(y, m, v)

    def log_func(logparams, ms, vcs):
        return func(numpy.exp(logparams), m, v)

    # First calculate the observed hessian.
    if not log:
        hess = -get_hess(func, p0, eps, args=[ms, vcs])
    else:
        hess = -get_hess(log_func, numpy.log(p0), eps, args=[ms, vcs])

    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = numpy.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = numpy.zeros((len(p0),1))
    for ii, (m, v) in enumerate(zip(all_boot[0], all_boot[1])):
        #boot = LDstats(boot)
        if not log:
            grad_temp = get_grad(func, p0, eps, args=[m, v])
        else:
            grad_temp = get_grad(log_func, numpy.log(p0), eps,
                                 args=[m, v])
        J_temp = numpy.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J/len(all_boot[0])
    cU = cU/len(all_boot[0])

    # G = H*J^-1*H
    J_inv = numpy.linalg.inv(J)
    godambe = numpy.dot(numpy.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


func_calls = 0
def GIM_uncert(model_func, all_boot, p0, ms, vcs, log=False,
               eps=0.01, return_GIM=False,
               r_edges=None, normalization=1, pass_Ne=False,
               statistics=None):
    """
    Parameter uncertainties from Godambe Information Matrix (GIM)

    Returns standard deviations of parameter values.

    model_func: Model function
    all_boot: List of bootstrap LD stats [[m0, m1, m2], [v0, v1, v2]]
    p0: Best-fit parameters for model_func, with inferred Ne in last entry of
        parameter list.
    ms, vcs: Original means and covariances of statistics from data.
    eps: Fractional stepsize to use when taking finite-difference derivatives.
         Note that if eps*param is < 1e-6, then the step size for that parameter
         will simply be eps, to avoid numerical issues with small parameter
         perturbations.
    log: If True, assume log-normal distribution of parameters. Returned values
         are then the standard deviations of the *logs* of the parameter values,
         which can be interpreted as relative parameter uncertainties.
    return_GIM: If true, also return the full GIM.
    
    Specific for LD stats computations
    r_edges: 
    normalization:
    pass_Ne: 
    """    
    assert statistics is not None, "need to pass statistics = ..."
        
    def pass_func(params, statistics):
        global func_calls
        func_calls += 1
        print(f"called {func_calls} times")
        print(params)
        rho = 4*params[-1]*r_edges
        if pass_Ne:
            y = Inference.bin_stats(model_func, params, rho=rho)
        else:
            y = Inference.bin_stats(model_func, params[:-1], rho=rho)
        y = Inference.sigmaD2(y, normalization=normalization)
        y = Inference.remove_nonpresent_statistics(y, statistics)
        return y

    GIM, H, J, cU = get_godambe(pass_func, all_boot, p0, ms, vcs, eps, statistics, log=log)

    uncerts = numpy.sqrt(numpy.diag(numpy.linalg.inv(GIM)))
    if not return_GIM:
        return uncerts
    else:
        return uncerts, GIM

