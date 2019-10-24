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
    Y = LDstats(Y, num_pops=1)
    Y = Y.split(1)
    Y.pop_ids = pop_ids
    return Y

def split_mig(params, rho=None, theta=0.001, pop_ids=None):
    """
    params: (nu1, nu2, T, m)
    
    Split into two populations of specifed size, with migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    m: Migration rate between populations (2*Na*m)
    """
    nu1, nu2, T, m = params
    
    Y = snm(rho=rho, theta=theta)
    Y.integrate([nu1,nu2], T, rho=rho, theta=theta, m=[[0,m],[m,0]])
    Y.pop_ids = pop_ids
    return Y

