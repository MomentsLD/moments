import numpy as np
import math
import os,sys

from moments.LD import Numerics

import copy
import moments

from scipy.special import gammaln
import scipy.optimize

"""
Adapted from moments/dadi to infer input parameters of demographic model
Usage is the same as moments.Inference, but inference using LD statistics requires a bit more for inputs
There are two options: run inference with LD stats alone, or LD+AFS
If we are using LD stats alone, data = [means, varcovs], a list of statistics means and the bootstrapped variance-covariance matrix
If we use LD+AFS, data = [means, varcovs, fs]
To use the frequency spectrum in the inference, we set the flag use_afs=True

"""

_counter = 0

def multivariate_normal_pdf(x,mu,Sigma):
    p = len(x)
    return np.sqrt(np.linalg.det(Sigma)/(2*math.pi)**p) * np.exp( -1./2 * 
                        np.dot( np.dot( (x-mu).transpose() , np.linalg.inv(Sigma) ) , x-mu )
                        )

def ll(x,mu,Sigma):
    """
    x = data
    mu = model function output
    Sigma = variance-covariance matrix
    """
    return -1./2 * np.dot( np.dot( (x-mu).transpose() , np.linalg.inv(Sigma) ) , x-mu ) #- len(x)*np.pi - 1./2*np.log(np.linalg.det(Sigma)) 

def ll_over_bins(xs,mus,Sigmas):
    """
    xs = list of data arrays
    mus = list of model function output arrays
    Sigmas = list of var-cov matrices
    Lists must be in the same order
    Each bin is assumed to be independent, so we call ll(x,mu,Sigma) for each bin
    """
    it = iter([xs,mus,Sigmas])
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Lists of data, means, and varcov matrices must be the same length')
    ll_val = 0
    for ii in range(len(xs)):
        ll_val += ll(xs[ii],mus[ii],Sigmas[ii])
    return ll_val

_out_of_bounds_val = -1e12
def _object_func(params, ns, model_func, means, varcovs, fs=None, rhos=[0],
                 order=2, theta=None, Leff=None, ism=True, corrected=True,
                 lower_bound=None, upper_bound=None,
                 verbose=0, func_args=[], func_kwargs={},
                 fixed_params=None, multinom=False, use_afs=False,
                 output_stream=sys.stdout):
    global _counter
    _counter += 1
    
    # Deal with fixed parameters
    params_up = _project_params_up(params, fixed_params)
    
    # Check our parameter bounds
    if lower_bound is not None:
        for pval,bound in zip(params_up, lower_bound):
            if bound is not None and pval < bound:
                return -_out_of_bounds_val
    if upper_bound is not None:
        for pval,bound in zip(params_up, upper_bound):
            if bound is not None and pval > bound:
                return -_out_of_bounds_val
    
    ### how to multinom with afs + LD
    if multinom == True:
        theta = params_up[-1]
    
    all_args = [params_up] + list(func_args)
    ## first get ll of afs
    if use_afs == True:
        model = Leff * model_func[1](all_args[0],[ns])
        ll_afs = moments.Inference.ll(fs,model)
    
    if multinom == True:
        all_args = [all_args[0][:-1]]

    ## next get ll for LD stats
    # need func_kwargs for each rho in rhos
    func_kwargs_list = []
    for rho in rhos:
        func_kwargs_list.append( {'theta':theta, 'ns':ns, 'rho':rho, 'corrected':corrected, 'ism':ism} )
    
    sorted_rhos = np.sort(rhos)
    mid_rhos = (rhos[1:]+rhos[:-1])/2.
    func_kwargs_list_mids = []
    for rho in mid_rhos:
        func_kwargs_list_mids.append( {'theta':theta, 'ns':ns, 'rho':rho, 'corrected':corrected, 'ism':ism} )
    
    if use_afs == True: # we adjust varcovs and means to remove sigma statistics
        # we don't want the sigma statistics, since they are just summaries of the frequency spectrum
        names = Numerics.moment_names_onepop(order)
        inds_to_remove = [names.index('1_s{0}'.format(ii)) for ii in range(1,order/2+1)]
    else:
        inds_to_remove = []

    stats = []
    for func_kwargs_rho in func_kwargs_list:
        temp_stats = model_func[0](order, *all_args, **func_kwargs_rho)
        temp_stats = temp_stats[:-1] # last value is 1
        stats.append(np.delete(temp_stats,inds_to_remove))
    
    stats_mid = []
    for func_kwargs_rho in func_kwargs_list_mids:
        temp_stats = model_func[0](order, *all_args, **func_kwargs_rho)
        temp_stats = temp_stats[:-1] # last value is 1
        stats_mid.append(np.delete(temp_stats,inds_to_remove))
    
    ## rhos are the bin edges, so we used trapezoid to approx stats for each bin
    #trap_stats = []
    #for ii in range(len(stats)-1):
    #    trap_stats.append((stats[ii] + stats[ii+1])/2.)
    
    # turns out trapezoid isn't accurate enough - leads to bias in inference
    # Simpson's rule should do much better
    simp_stats = []
    for ii in range(len(stats)-1):
        simp_stats.append((stats[ii] + 4*stats_mid[ii] + stats[ii+1])/6.)

    ## result in ll from afs plus ll from rho bins
    if use_afs == True:
        result = ll_afs + ll_over_bins(means, simp_stats, varcovs)
    else:
        result = ll_over_bins(means, simp_stats, varcovs)
        
    # Bad result
    if np.isnan(result):
        result = _out_of_bounds_val
    
    if (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        output_stream.write('%-8i, %-12g, %s%s' % (_counter, result, param_str,
                                                   os.linesep))

    return -result

def _object_func_log(log_params, *args, **kwargs):
    return _object_func(np.exp(log_params), *args, **kwargs)

def optimize_log_fmin(p0, ns, data, model_func, rhos=[0],
                 order=2, theta=None, Leff=None, ism=True, corrected=True,
                 lower_bound=None, upper_bound=None, verbose=0,
                 func_args=[], func_kwargs={}, fixed_params=None, 
                 multinom=False, use_afs=False):
    """
    p0 = initial guess
    ns = sample size (number of haplotypes) - we need this to correct for sampling bias
    data = [means, varcovs, fs (optional, use if use_afs=True)]
    means = list of mean statistics matching rhos in func_kwargs
    varcovs = list of varcov matrices matching means
    model_func = demographic model to compute statistics for a given rho
        If we are using AFS, it's a list of the two models [LD, AFS]
        If it's LD stats alone, it's just a single LD model (still as a list)
    order = the single-population order of D-statistics
    theta = NOTE!! this is population scaled per base mutation rate (4Ne*mu, not 4Ne*mu*L)
    func_kwargs = need 'ns', 'rhos', and 'theta'
    """
    # update this to write to file, now just prints to stdout
    output_stream = sys.stdout
    
    means = data[0]
    varcovs = data[1]
    if use_afs == True:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError("if use_afs=True, need to pass frequency spectrum in data=[means,varcovs,fs]")
    else:
        fs = None
    
    if multinom == False and theta == None:
        raise ValueError("if multinom is False, need to specify theta")
    
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    
    if use_afs == True: # we adjust varcovs and means to remove sigma statistics
        # we don't want the sigma statistics, since they are just summaries of the frequency spectrum
        names = Numerics.moment_names_onepop(order)
        inds_to_remove = [names.index('1_s{0}'.format(ii)) for ii in range(1,order/2+1)]
        for ii in range(len(vcs)):
            vcs[ii] = np.delete(vcs[ii], inds_to_remove, axis=0)
            vcs[ii] = np.delete(vcs[ii], inds_to_remove, axis=1)
        for ii in range(len(ms)):
            ms[ii] = np.delete(ms[ii], inds_to_remove)
    
    args = (ns, model_func, ms, vcs, fs, rhos,
            order, theta, Leff, ism, corrected,
            lower_bound, upper_bound, 
            verbose, func_args, func_kwargs,
            fixed_params, multinom, use_afs,
            output_stream)
    
    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin(_object_func_log, np.log(p0), args=args, full_output=True, disp=False)
    
    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(np.exp(xopt), fixed_params)
    
    return xopt

def _project_params_down(pin, fixed_params):
    """
    Eliminate fixed parameters from pin.
    """
    if fixed_params is None:
        return pin

    if len(pin) != len(fixed_params):
        raise ValueError('fixed_params list must have same length as input '
                         'parameter array.')

    pout = []
    for ii, (curr_val,fixed_val) in enumerate(zip(pin, fixed_params)):
        if fixed_val is None:
            pout.append(curr_val)

    return np.array(pout)

def _project_params_up(pin, fixed_params):
    """
    Fold fixed parameters into pin.
    """
    if fixed_params is None:
        return pin

    if np.isscalar(pin):
        pin = [pin]

    pout = np.zeros(len(fixed_params))
    orig_ii = 0
    for out_ii, val in enumerate(fixed_params):
        if val is None:
            pout[out_ii] = pin[orig_ii]
            orig_ii += 1
        else:
            pout[out_ii] = fixed_params[out_ii]
    return pout

