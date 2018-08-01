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


def equilibrium(ns, rho=None, theta=1.0, gamma=None, h=0.5, sel_params=None):
    """
    Compute or load the equilibrium two locus frequency spectrum
    """
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0
    
    if gamma==None:
        gamma=0.0
    
    # fetch from cache if neutral (cache only neutral spectra for the moment)
    eq_name = 'tlfs_ns{0}_rho{1}_theta{2}.fs'.format(ns,rho,theta)
    eq_name = os.path.join(cache_path, eq_name)
    try:
        F = moments.TwoLocus.TLSpectrum.from_file(eq_name)
    except IOError:
        F = np.zeros((ns+1,ns+1,ns+1))
        F = moments.TwoLocus.Integration.integrate(F, 1.0, 40.0, rho=rho, theta=theta, 
                                gamma=gamma, h=h, sel_params=sel_params)
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

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    nu_func = lambda t: nuB * np.exp(np.log(nuF/nuB) * t/T)
    F.integrate(nu_func, T, rho=rho, theta=theta, gamma=gamma, h=h, sel_params=sel_params)
    return F

