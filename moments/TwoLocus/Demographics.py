import numpy as np
import moments.TwoLocus.Integration
import moments.TwoLocus
import os

"""
Demographic models for two locus model (all single population)
Two-epoch
Three-epoch
Growth
Bottlegrowth
"""

# Cache equilibrium spectra in ~/.moments/TwoLocus_cache by default
def set_cache_path(path='~/.moments/TwoLocus_cache'):
    """
    Set directory in which demographic equilibrium phi spectra will be cached.

    The collection of cached spectra can get large, so it may be helpful to
    store them outside the user's home directory.
    """
    global cache_path
    cache_path = os.path.expanduser(path)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

cache_path=None
set_cache_path()

def make_floats(params):
    """
    pass list of params, return floats of those params (for caching)
    """
    if params is None:
        return None
    if hasattr(params, "__len__"):
        return [float(p) for p in params]
    else:
        return float(params)

def equilibrium(ns, rho=None, theta=1.0, gamma=None, h=0.5, sel_params=None):
    """
    Compute or load the equilibrium two locus frequency spectrum
    gamma and h, if set, are only for selection at the A/a locus
    sel_params, which are the (additive) selection coefficients for haplotypes AB, Ab, and aB
        so that sel_params = (sAB, sA, sB)
        If sAB = sA + sB, no epistatic interaction
    """
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0
    
    if gamma==None:
        gamma=0.0
    
    gamma = make_floats(gamma)
    rho = make_floats(rho)
    theta = make_floats(theta)
    sel_params = make_floats(sel_params)
    
    # fetch from cache if neutral (cache only neutral spectra for the moment)
    if gamma == 0.0 and (sel_params == None or (sel_params[0] == 0.0 and sel_params[1] == 0.0 and sel_params[2] == 0.0)):
        eq_name = 'tlfs_ns{0}_rho{1}_theta{2}.fs'.format(ns,rho,theta)
        eq_name = os.path.join(cache_path, eq_name)
    elif gamma != 0.0: # cache for gammaA, hA
        eq_name = 'tlfs_ns{0}_rho{1}_theta{2}_gammaA{3}_hA{4}.fs'.format(ns,rho,theta,gamma,h)
        eq_name = os.path.join(cache_path, eq_name)
    elif sel_params != None:
        eq_name = 'tlfs_ns{0}_rho{1}_theta{2}_sel_{3}_{4}_{5}.fs'.format(ns,rho,theta,sel_params[0],sel_params[1],sel_params[2])
        eq_name = os.path.join(cache_path, eq_name)
    
    try:
        F = moments.TwoLocus.TLSpectrum.from_file(eq_name)
    except IOError:
        F = np.zeros((ns+1,ns+1,ns+1))
        ### I would rather compute the equilibrium state by inverting the transition matrix
        ### so long as it is not poory conditioned (note 7/16)
        F = moments.TwoLocus.Integration.integrate(F, 1.0, 40.0, rho=rho, theta=theta, 
                                gamma=gamma, h=h, sel_params=sel_params, dt=0.001)
        F = moments.TwoLocus.TLSpectrum(F)
        F.to_file(eq_name)
    return F

def two_epoch(params, ns, rho=None, theta=1.0, gamma=None, h=0.5, sel_params=None):
    """
    params = (nu,T)
    nu - size change
    T - time in past size change occured
    """
    nu,T = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0
    
    if gamma==None:
        gamma=0.0

    gamma = make_floats(gamma)
    rho = make_floats(rho)
    theta = make_floats(rho)
    sel_params = make_floats(sel_params)
    
    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    F.integrate(nu, T, rho=rho, theta=theta, 
                gamma=gamma, h=h, sel_params=sel_params)
    return F

def three_epoch(params, ns, rho=None, theta=1.0, gamma=None, h=0.5, sel_params=None):
    """
    params = (nu1,nu2,T1,T2)
    nu - size change
    T - time in past size change occured
    """
    nu1,nu2,T1,T2 = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0
    
    if gamma==None:
        gamma=0.0
    
    gamma = make_floats(gamma)
    rho = make_floats(rho)
    theta = make_floats(rho)
    sel_params = make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    F.integrate(nu1, T1, rho=rho, theta=theta, 
                gamma=gamma, h=h, sel_params=sel_params)
    F.integrate(nu2, T2, rho=rho, theta=theta, 
                gamma=gamma, h=h, sel_params=sel_params)
    return F

def growth(params, ns, rho=None, theta=1.0, gamma=None, h=0.5, sel_params=None):
    """
    exponential growth or decay model
    params = (nu,T)
    nu - final size
    T - time in past size changes begin
    """
    nu,T = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0
    
    if gamma==None:
        gamma=0.0

    gamma = make_floats(gamma)
    rho = make_floats(rho)
    theta = make_floats(rho)
    sel_params = make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    nu_func = lambda t: np.exp(np.log(nu) * t/T)
    F.integrate(nu_func, T, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    return F

def bottlegrowth(params, ns, rho=None, theta=1.0, gamma=None, h=0.5, sel_params=None):
    """
    exponential growth or decay model
    params = (nuB,nuF,T)
    nuB - bottleneck size
    nu - final size
    T - time in past size changes begin
    """
    nuB,nuF,T = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0
    
    if gamma==None:
        gamma=0.0

    gamma = make_floats(gamma)
    rho = make_floats(rho)
    theta = make_floats(rho)
    sel_params = make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    nu_func = lambda t: nuB * np.exp(np.log(nuF/nuB) * t/T)
    F.integrate(nu_func, T, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    return F

