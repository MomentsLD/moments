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
Usage is the same as moments.Inference, but inference using LD statistics 
requires a bit more for inputs
There are two options: run inference with LD stats alone, or LD+AFS
If we are using LD stats alone, data = [means, varcovs], a list of statistics 
means and the bootstrapped variance-covariance matrix
If we use LD+AFS, data = [means, varcovs, fs]
To use the frequency spectrum in the inference, we set the flag use_afs=True

"""

_counter = 0

def multivariate_normal_pdf(x,mu,Sigma):
    p = len(x)
    return np.sqrt(np.linalg.det(Sigma)/(2*math.pi)**p) * np.exp( -1./2 * 
                        np.dot( np.dot( (x-mu).transpose() , 
                                np.linalg.inv(Sigma) ) , x-mu ) )

def ll(x,mu,Sigma):
    """
    x = data
    mu = model function output
    Sigma = variance-covariance matrix
    """
    return -1./2 * np.dot( np.dot( (x-mu).transpose() , 
                            np.linalg.inv(Sigma) ) , x-mu ) 
                            #- len(x)*np.pi - 1./2*np.log(np.linalg.det(Sigma)) 

def ll_over_bins(xs,mus,Sigmas):
    """
    xs = list of data arrays
    mus = list of model function output arrays
    Sigmas = list of var-cov matrices
    Lists must be in the same order
    Each bin is assumed to be independent, so we call ll(x,mu,Sigma) 
      for each bin
    """
    it = iter([xs,mus,Sigmas])
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('Lists of data, means, and varcov matrices must be the same length')
    ll_vals = []
    for ii in range(len(xs)):
        ll_vals.append(ll(xs[ii],mus[ii],Sigmas[ii]))
    ll_val = np.sum(ll_vals)
    return ll_val

_out_of_bounds_val = -1e12
def _object_func(params, ns, model_func, means, varcovs, fs=None,
                 rhos=[0], rs = None, het_model=None,
                 order=2, theta=None, u=None, Ne=None, pass_Ne=False,
                 Leff=None, ism=True, corrected=True,
                 lower_bound=None, upper_bound=None,
                 verbose=0, flush_delay=0, func_args=[], func_kwargs={},
                 fixed_params=None, multinom=False, fixed_theta=False, 
                 use_afs=False, genotypes=False, inds_to_remove=[], 
                 multipop=False, multipop_stats=None,
                 fixed_rho=True, fixed_u=True,
                 corr_mu=False,
                 output_stream=sys.stdout):
    global _counter
    _counter += 1
    
    # ns is [nsLD, nsFS]
    nsLD, nsFS = ns
    
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
    
    if corr_mu == True:
        mus = params_up[-2:]
        Ne = params_up[-3]
        p = params_up[-4]
        lam = params_up[-5]
        F0 = params_up[-6]
        mu_low = mus[0]
        mu_high = mus[1]
        mu_ave = mu_low * p + mu_high * (1-p)
        theta = 4*Ne*mu_ave
        all_args = [params_up] + list(func_args)
        all_args = [all_args[0][:-6]]
        rhos = [4*Ne*r for r in rs]
    else:
        if fixed_rho == False:
            ## rhos are r-values from recomb map
            if fixed_u == False:
                Ne, u = params_up[-2:]
                theta = 4*Ne*u
                all_args = [params_up] + list(func_args)
                all_args = [all_args[0][:-2]]
            else:
                Ne = params_up[-1]
                theta = 4*Ne*u
                all_args = [params_up] + list(func_args)
                if pass_Ne == False:
                    all_args = [all_args[0][:-1]]
                else:
                    all_args = [all_args[0]]
            rhos = [4*Ne*r for r in rs]
        else:
            if fixed_theta == False:
                theta = params_up[-1]
                all_args = [params_up] + list(func_args)
                all_args = [all_args[0][:-1]]
            else:
                all_args = [params_up] + list(func_args)

    ## first get ll of afs
    if use_afs == True:
        model = Leff * theta * model_func[1](all_args[0],nsFS)
        if fs.folded:
            model = model.fold()
        if multinom == True:
            ll_afs = moments.Inference.ll_multinom(model,fs)
        else:
            ll_afs = moments.Inference.ll(model,fs)
    
    ## next get ll for LD stats
    # need func_kwargs for each rho in rhos
    func_kwargs_list = []
    for rho in rhos:
        func_kwargs_list.append( {'order':order, 'theta':theta, 'ns':nsLD, 'rho':rho, 
                                        'corrected':corrected, 'ism':ism, 'genotypes':genotypes} )
    
    sorted_rhos = np.sort(rhos)
    mid_rhos = (sorted_rhos[1:]+sorted_rhos[:-1])/2.
    func_kwargs_list_mids = []
    for rho in mid_rhos:
        func_kwargs_list_mids.append( {'order':order, 'theta':theta, 'ns':nsLD, 'rho':rho, 
                                        'corrected':corrected, 'ism':ism, 'genotypes':genotypes} )

    stats = []
    for func_kwargs_rho in func_kwargs_list:
        temp_stats = model_func[0](*all_args, **func_kwargs_rho)
        rho = func_kwargs_rho['rho']
        if corr_mu == True:
            F = F0 * np.exp(-lam * rho)
        if multipop_stats == None:
            stats.append(np.delete(temp_stats,inds_to_remove))
        else:
            if corr_mu == False:
                stats.append(temp_stats[0])
            if corr_mu == True:
                stats.append( (mu_low/mu_ave)**2 * F * p * temp_stats[0] + (mu_high/mu_ave)**2 * F * (1-p) * temp_stats[0] + (1-F) * temp_stats[0] )
    
    stats_mid = []
    for func_kwargs_rho in func_kwargs_list_mids:
        temp_stats = model_func[0](*all_args, **func_kwargs_rho)
        rho = func_kwargs_rho['rho']
        if corr_mu == True:
            F = F0 * np.exp(-lam * rho)
        if multipop_stats == None:
            stats_mid.append(np.delete(temp_stats,inds_to_remove))
        else:
            if corr_mu == False:
                stats_mid.append(temp_stats[0])
            if corr_mu == True:
                stats_mid.append( (mu_low/mu_ave)**2 * F * p * temp_stats[0] + (mu_high/mu_ave)**2 * F * (1-p) * temp_stats[0] + (1-F) * temp_stats[0] )
    
    if het_model == None:
        one_locus_stats = temp_stats[1]
    else:
        one_locus_stats = het_model(params_up, theta=theta)
    
    ## rhos are the bin edges, so we used trapezoid to approx stats for each bin
    #trap_stats = []
    #for ii in range(len(stats)-1):
    #    trap_stats.append((stats[ii] + stats[ii+1])/2.)
    
    # turns out trapezoid isn't too accurate - leads to bias in inference
    # Simpson's rule should perform much better
    simp_stats = []
    for ii in range(len(stats)-1):
        simp_stats.append((stats[ii] + 4*stats_mid[ii] + stats[ii+1])/6.)
    
    if multipop == True and use_afs == False:
        simp_stats.append(one_locus_stats)
    
    ## result in ll from afs plus ll from rho bins
    if use_afs == True:
        result = ll_afs + ll_over_bins(means, simp_stats, varcovs)
        #print ll_afs, ll_over_bins(means, simp_stats, varcovs)
    else:
        result = ll_over_bins(means, simp_stats, varcovs)
    
    # Bad result
    if np.isnan(result):
        result = _out_of_bounds_val
        
    if (verbose > 0) and (_counter % verbose == 0):
        param_str = 'array([%s])' % (', '.join(['%- 12g'%v for v in params_up]))
        output_stream.write('%-8i, %-12g, %s%s' % (_counter, result, param_str,
                                                   os.linesep))
        moments.Misc.delayed_flush(delay=flush_delay)
    
    return -result

def _object_func_log(log_params, *args, **kwargs):
    return _object_func(np.exp(log_params), *args, **kwargs)

def optimize_log_fmin(p0, ns, data, model_func, het_model=None, rhos=[0], rs=None,
                 order=2, theta=None, u=None, Ne=None, pass_Ne=False,
                 Leff=None, ism=True, corrected=True,
                 lower_bound=None, upper_bound=None, 
                 verbose=0, flush_delay=0.5,
                 func_args=[], func_kwargs={}, fixed_params=None, 
                 multinom=False, fixed_theta=False, use_afs=False, 
                 genotypes=False, num_pops=1, 
                 multipop=False, multipop_stats=None, 
                 fixed_rho=True, fixed_u=True,
                 corr_mu=False):
    """
    p0 = initial guess (demography parameters + theta)
    ns = sample size (number of haplotypes) can be passed as single value or [nsLD,nsFS]
         if different sample sizes were used for computing LD stats and the frequency spectrum
    data = [means, varcovs, fs (optional, use if use_afs=True)]
    means = list of mean statistics matching rhos in func_kwargs
    varcovs = list of varcov matrices matching means
    model_func = demographic model to compute statistics for a given rho
        If we are using AFS, it's a list of the two models [LD, AFS]
        If it's LD stats alone, it's just a single LD model (still as a list)
    order = the single-population order of D-statistics
    theta = this is population scaled per base mutation rate (4Ne*mu, not 4Ne*mu*L)
    
    multinom: If True, we allow separate effective theta for the frequency spectrum and 
                theta is only fit to the lD data. If False, the same that scales both LD
                statistics and the fs
    fixed_theta: If True, theta is fixed to input theta. Otherwise a guess is passed  (as in multinom in moments inference) (I know this is bad)
    Leff: effective length of genome from which the fs was generated
    
    To Do: make this flexible to be able to handle multipopulation inference
    """
    output_stream = sys.stdout
    
    if hasattr(ns, '__len__') == False: # this works for single populations
        ns = [[ns],[ns]]
    
    means = data[0]
    varcovs = data[1]
    if use_afs == True:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError("if use_afs=True, need to pass frequency spectrum in data=[means,varcovs,fs]")
    else:
        fs = None
        
    if fixed_theta == True and theta == None:
        raise ValueError("if multinom is False, need to specify theta")
    
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    
    inds_to_remove = []
    
    args = (ns, model_func, ms, vcs, fs, rhos, rs, het_model,
            order, theta, u, Ne, pass_Ne, Leff, ism, corrected,
            lower_bound, upper_bound, 
            verbose, flush_delay, func_args, func_kwargs,
            fixed_params, multinom, fixed_theta, 
            use_afs, genotypes, inds_to_remove, 
            multipop, multipop_stats, fixed_rho, fixed_u,
            corr_mu,
            output_stream)
    
    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin(_object_func_log, np.log(p0), args=args, full_output=True, disp=False)
    
    xopt, fopt, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(np.exp(xopt), fixed_params)
    
    return xopt, fopt

def optimize_log_powell(p0, ns, data, model_func, het_model=None, rhos=[0], rs=None,
                 order=2, theta=None, u=None, Ne=None, pass_Ne=False,
                 Leff=None, ism=True, corrected=True,
                 lower_bound=None, upper_bound=None, 
                 verbose=0, flush_delay=0.5,
                 func_args=[], func_kwargs={}, fixed_params=None, 
                 multinom=False, fixed_theta=False, use_afs=False, 
                 genotypes=False, num_pops=1, 
                 multipop=False, multipop_stats=None, 
                 fixed_rho=True, fixed_u=True,
                 corr_mu=False):
    """
    p0 = initial guess (demography parameters + theta)
    ns = sample size (number of haplotypes) can be passed as single value or [nsLD,nsFS]
         if different sample sizes were used for computing LD stats and the frequency spectrum
    data = [means, varcovs, fs (optional, use if use_afs=True)]
    means = list of mean statistics matching rhos in func_kwargs
    varcovs = list of varcov matrices matching means
    model_func = demographic model to compute statistics for a given rho
        If we are using AFS, it's a list of the two models [LD, AFS]
        If it's LD stats alone, it's just a single LD model (still as a list)
    order = the single-population order of D-statistics
    theta = this is population scaled per base mutation rate (4Ne*mu, not 4Ne*mu*L)
    
    multinom: If True, we allow separate effective theta for the frequency spectrum and 
                theta is only fit to the lD data. If False, the same that scales both LD
                statistics and the fs
    fixed_theta: If True, theta is fixed to input theta. Otherwise a guess is passed  (as in multinom in moments inference) (I know this is bad)
    Leff: effective length of genome from which the fs was generated
    
    To Do: make this flexible to be able to handle multipopulation inference
    """
    output_stream = sys.stdout
    
    if hasattr(ns, '__len__') == False: # this works for single populations
        ns = [[ns],[ns]]
    
    means = data[0]
    varcovs = data[1]
    if use_afs == True:
        try:
            fs = data[2]
        except IndexError:
            raise ValueError("if use_afs=True, need to pass frequency spectrum in data=[means,varcovs,fs]")
    else:
        fs = None
        
    if fixed_theta == True and theta == None:
        raise ValueError("if multinom is False, need to specify theta")
    
    ms = copy.copy(means)
    vcs = copy.copy(varcovs)
    
    inds_to_remove = []
    
    args = (ns, model_func, ms, vcs, fs, rhos, rs, het_model,
            order, theta, u, Ne, pass_Ne, Leff, ism, corrected,
            lower_bound, upper_bound, 
            verbose, flush_delay, func_args, func_kwargs,
            fixed_params, multinom, fixed_theta, 
            use_afs, genotypes, inds_to_remove, 
            multipop, multipop_stats, fixed_rho, fixed_u,
            corr_mu,
            output_stream)
    
    p0 = _project_params_down(p0, fixed_params)
    outputs = scipy.optimize.fmin_powell(_object_func_log, np.log(p0), args=args, full_output=True, disp=False)
    
    xopt, fopt, direc, iter, funcalls, warnflag = outputs
    xopt = _project_params_up(np.exp(xopt), fixed_params)
    
    return xopt, fopt


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

