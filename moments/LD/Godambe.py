"""
Parameter uncertainties and likelihood ratio tests using Godambe information.
Adapted from Coffman et al (2016) to deal with LD statistics instead of frequency spectra.
"""
import numpy as np

from moments.LD import Inference
from moments.LD.LDstats_mod import LDstats

def hessian_elem(func, f0, p0, ii, jj, eps, args=()):
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
    pwork = np.array(p0, copy=True, dtype=float)
    if ii == jj:
        if pwork[ii] != 0:
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)
            
            pwork[ii] = p0[ii] - eps[ii]
            fm = func(pwork, *args)
            
            element = (fp - 2*f0 + fm)/eps[ii]**2
        if pwork[ii] == 0:
            pwork[ii] = p0[ii] + 2*eps[ii]
            fpp = func(pwork, *args)
            
            pwork[ii] = p0[ii] + eps[ii]
            fp = func(pwork, *args)

            element = (fpp - 2*fp + f0)/eps[ii]**2
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
    args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = np.empty([len(p0)])
    for i, pval in enumerate(p0):
        if pval != 0:
            eps[i] = eps_in*pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    f0 = func(p0, *args)
    hess = np.empty((len(p0), len(p0)))
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            hess[ii][jj] = hessian_elem(func, f0, p0, ii, jj, eps, args=args)
            hess[jj][ii] = hess[ii][jj]
    return hess

def get_grad(func, p0, eps, args=()):
    """
    Calculate gradient vector
    
    func: Model function
    p0: Parameters for func
    eps: Fractional stepsize to use when taking finite-difference derivatives
    args: Additional arguments to func
    """
    # Calculate step sizes for finite-differences.
    eps_in = eps
    eps = np.empty([len(p0)])
    for i, pval in enumerate(p0):
        if pval != 0:
            eps[i] = eps_in * pval
        else:
            # Account for parameters equal to zero
            eps[i] = eps_in

    grad = np.empty([len(p0), 1])
    for ii in range(len(p0)):
        pwork = np.array(p0, copy=True, dtype=float)

        if p0[ii] != 0:
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

def get_godambe(func_ex, all_boot, p0, data, vcs, args=(), eps=0.01, 
                log=False, just_hess=False):
    """
    Godambe information and Hessian matrices

    NOTE: Assumes that last parameter in p0 is N.

    func_ex: Model function
    all_boot: List of bootstrap LD stats
    p0: Best-fit parameters for func_ex
    data: Original data LD stats
    vcs: Original variance covariance matrix for likelihood calculation
    args: Arguments to pass to func_ex (such as theta, rho, corrected, ns, ...)
    eps: Fractional stepsize to use when taking finite-difference derivatives
    log: If True, calculate derivatives in terms of log-parameters
    just_hess: If True, only evaluate and return the Hessian matrix
    """
    # Cache evaluations of the LD stats inside our hessian/J 
    # evaluation function
    cache = {}
    def func(params, data):
        key = (tuple(params))
        if key not in cache:
            cache[key] = func_ex(params, *args)
        stats = cache[key] 
        return Inference.ll_over_bins(data, stats, vcs)
    def log_func(logparams, data):
        return func(np.exp(logparams), data)

    # First calculate the observed hessian
    if not log:
        hess = -get_hess(func, p0, eps, args=[data])
    else:
        hess = -get_hess(log_func, np.log(p0), eps, args=[data])

    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = np.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = np.zeros((len(p0), 1))
    for ii, boot in enumerate(all_boot):
        if not log:
            grad_temp = get_grad(func, p0, eps, args=[boot])
        else:
            grad_temp = get_grad(log_func, np.log(p0), eps, args=[boot])

        J_temp = np.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = np.linalg.inv(J)
    godambe = np.dot(np.dot(hess, J_inv), hess)
    return godambe, hess, J, cU

def GIM_uncert(func_ex, all_boot, p0, data, vcs, args=(), log=False,
               eps=0.01, return_GIM=False):
    """
    Parameter uncertainties from Godambe Information Matrix (GIM)

    Returns standard deviations of parameter values.

    NOTE: Assumes that last parameter in p0 is N.
    
    func_ex: Model function
    all_boot: List of bootstrap LD stats
    p0: Best-fit parameters for func_ex
    data: Original data LD stats
    vcs: Original variance covariance matrix for likelihood calculation
    args: Arguments to pass to func_ex (such as theta, rho, corrected, ns, ...)
    eps: Fractional stepsize to use when taking finite-difference derivatives
    log: If True, assume log-normal distribution of parameters. Returned values 
         are then the standard deviations of the *logs* of the parameter values,
         which can be interpreted as relative parameter uncertainties.
    return_GIM: If true, also return the full GIM.
    """
    GIM, H, J, cU = get_godambe(func_ex, all_boot, p0, data, vcs, args, eps, log)
    uncerts = np.sqrt(np.diag(np.linalg.inv(GIM)))
    if not return_GIM:
        return uncerts
    else:
        return uncerts, GIM

