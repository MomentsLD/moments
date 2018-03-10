import numpy as np
from moments.LD import Numerics
from moments.LD import Corrections
from moments.LD.LDstats_mod import LDstats

def snm(order=2, rho=0, theta=0.0008, ns=[200,200], corrected=False, genotypes=False):
    """
    Equilibrium neutral model
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for sampling bias in sample
               size ns    
    Standard neutral model, with populations never diverging
    """
    if order != 2:
        raise ValueError("We can only run multipopulation demographies for order 2 statistics.")
    
    n1,n2 = ns
    
    y = Numerics.root_equilibrium(rho, theta)
    y = LDstats(y, num_pops=1, order=order)
    
    y = y.split(1)
    
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_multipop(y, ns=(n1,n2), num_pops=2)
        else:
            return Corrections.corrected_multipop_genotypes(y, ns=(n1/2,n2/2), num_pops=2)
    else:
        return y

def split_mig(params, order=2, rho=0, theta=0.0008, ns=[200,200], corrected=False, genotypes=False):
    """
    params: (nu1, nu2, T, m)
    
    Split into two populations of specifed size, with migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    m: Migration rate between populations (2*Na*m)
    n1, n2: Sample sizes of resulting Spectrum.
    """
    if order != 2:
        raise ValueError("We can only run multipopulation demographies for order 2 statistics.")
    
    nu1, nu2, T, m = params
    
    n1,n2 = ns
    
    y = Numerics.root_equilibrium(rho, theta)
    y = LDstats(y, num_pops=1, order=order)
    
    y = y.split(1)
    y.integrate([nu1,nu2], T, rho=rho, theta=theta, m=[[0,m],[m,0]])
    
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_multipop(y, ns=(n1,n2), num_pops=2)
        else:
            return Corrections.corrected_multipop_genotypes(y, ns=(n1/2,n2/2), num_pops=2)
    else:
        return y

def IM(params, order=2, rho=0, theta=0.0008, ns=[200,200], corrected=False, genotypes=False):
    """
    params: (s, nu1, nu2, T, m12, m21)
    Isolation-with-migration model with exponential pop growth.

    s: Size of pop 1 after split. (Pop 2 has size 1-s.)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2 * Na * m12)
    m21: Migration from pop 1 to pop 2
    """
    if order != 2:
        raise ValueError("We can only run multipopulation demographies for order 2 statistics.")
    
    s, nu1, nu2, T, m12, m21 = params
    
    n1,n2 = ns
    
    y = Numerics.root_equilibrium(rho, theta)
    y = LDstats(y, num_pops=1, order=order)
    
    y = y.split(1)
    
    nu1_func = lambda t: s * (nu1/s)**(t/T)
    nu2_func = lambda t: (1-s) * (nu2/(1-s))**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]

    y.integrate(nu_func, T, rho=rho, theta=theta, m=[[0,m12],[m21,0]])
    
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_multipop(y, ns=(n1,n2), num_pops=2)
        else:
            return Corrections.corrected_multipop_genotypes(y, ns=(n1/2,n2/2), num_pops=2)
    else:
        return y

def IM_pre(params, order=2, rho=0, theta=0.0008, ns=[200,200], corrected=False, genotypes=False):
    """
    params = (nuPre, TPre, s, nu1, nu2, T, m12, m21)

    Isolation-with-migration model with exponential pop growth and a size change
    prior to split.

    nuPre: Size after first size change
    TPre: Time before split of first size change.
    s: Fraction of nuPre that goes to pop1. (Pop 2 has size nuPre*(1-s).)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    """
    if order != 2:
        raise ValueError("We can only run multipopulation demographies for order 2 statistics.")
    
    nuPre, TPre, s, nu1, nu2, T, m12, m21 = params
    
    n1,n2 = ns
    
    y = Numerics.root_equilibrium(rho, theta)
    y = LDstats(y, num_pops=1, order=order)
    
    y.integrate([nuPre], TPre, rho=rho, theta=theta)
    
    y = y.split(1)
    
    nu1_0 = nuPre * s
    nu2_0 = nuPre * (1-s)
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]

    y.integrate(nu_func, T, rho=rho, theta=theta, m=[[0,m12],[m21,0]])
    
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_multipop(y, ns=(n1,n2), num_pops=2)
        else:
            return Corrections.corrected_multipop_genotypes(y, ns=(n1/2,n2/2), num_pops=2)
    else:
        return y
