import numpy as np
from moments.LD import Demography
from moments.LD.LDstats_mod import LDstats

def snm(rho=None, theta=0.001, pop_ids=None):
    """
    Equilibrium neutral model
    rho: population-scaled recombination rate (4Nr), given as scalar or list of rhos
    theta: population-scaled mutation rate (4Nu)
    """
    Y = Demography.equilibrium(rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    return Y

def two_epoch(params, rho=None, theta=0.001, pop_ids=None):
    """
    Two epoch model
    params:  = (nu,T), where nu is the new population size, integrated for time T
    rho: population-scaled recombination rate (4Nr), given as scalar or list of rhos
    theta: population-scaled mutation rate (4Nu)
    """
    nu,T = params
    Y = Demography.equilibrium(rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    Y.integrate([nu], T, rho=rho, theta=theta)
    return Y

def three_epoch(params, rho=None, theta=0.001, pop_ids=None):
    """
    Three epoch model
    params:  = (nu1,nu2,T1,T2), where nus are the population size, integrated 
            for times T1 and T2
    rho: population-scaled recombination rate (4Nr), given as scalar or list of rhos
    theta: population-scaled mutation rate (4Nu)
    """
    nu1,nu2,T1,T2 = params
    Y = Demography.equilibrium(rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    Y.integrate([nu1], T1, rho=rho, theta=theta)
    Y.integrate([nu2], T2, rho=rho, theta=theta)
    return Y

def growth(params, order=2, rho=0, theta=0.001, pop_ids=None):
    """
    Exponential growth (or decay) model
    params: = (nuF,T), nu F is the final population size after time T (starting from nu=1)
    rho: population-scaled recombination rate (4Nr), given as scalar or list of rhos
    theta: population-scaled mutation rate (4Nu)
    """
    nuF,T = params
    Y = Demography.equilibrium(rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    nu_func = lambda t: [np.exp( np.log(nuF) *t/T)]
    Y.integrate(nu_func, T, rho=rho, theta=theta)
    return Y

def bottlegrowth(params, ns=200, rho=0, theta=0.001, pop_ids=None):    
    """
    Exponential growth (or decay) model after size change
    params: = (nuB,nuF,T), nu F is the final population size after time T, 
                starting from instantaneous population size change of nuB
    rho: population-scaled recombination rate (4Nr), given as scalar or list of rhos
    theta: population-scaled mutation rate (4Nu)
    """
    nuB,nuF,T = params
    Y = Demography.equilibrium(rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    nu_func = lambda t: [nuB * np.exp( np.log(nuF/nuB) *t/T)]
    Y.integrate(nu_func, T, rho=rho, theta=theta)
    return Y
