import numpy as np
from moments.LD import Numerics
from moments.LD import Corrections
from moments.LD.LDstats_mod import LDstats

def equilibrium(order=2, rho=0, theta=0.0008, ns=200, corrected=False, ism=False, genotypes=False):
    """
    Equilibrium neutral model
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for sampling bias in sample
               size ns
    ism: if True, we assume an infinite site model
         if False (default), we assume a reversible mutation model with equal forward
            and back mutation rates theta
    """
    y = Numerics.equilibrium(rho, theta, ism=ism, order=order)
    y = LDstats(y, num_pops=1, order=order)
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_onepop(y, n=ns, order=order)
        else:
            return Corrections.corrected_onepop_genotypes(y, n=ns/2, order=order)
    else:
        return y

def two_epoch(params, order=2, rho=0, theta=0.0008, ns=200, corrected=False, ism=False, genotypes=False):
    """
    Two epoch model
    params:  = (nu,T), where nu is the new population size, integrated for time T
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for sampling bias in sample
               size ns
    ism: if True, we assume an infinite site model
         if False (default), we assume a reversible mutation model with equal forward
            and back mutation rates theta
    """
    nu,T = params
    y = equilibrium(rho, theta, ism=ism, order=order)
    y = Numerics.integrate(y, T, rho=rho, theta=theta, nu=nu, order=order, dt=0.001, ism=ism)
    y = LDstats(y, num_pops=1, order=order)
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_onepop(y, n=ns, order=order)
        else:
            return Corrections.corrected_onepop_genotypes(y, n=ns/2, order=order)
    else:
        return y

def three_epoch(params, order=2, rho=0, theta=0.0008, ns=200, corrected=False, ism=False, genotypes=False):
    """
    Three epoch model
    params:  = (nu1,nu2,T1,T2), where nus are the population size, integrated 
            for times T1 and T2
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for sampling bias in sample
               size ns
    ism: if True, we assume an infinite site model
         if False (default), we assume a reversible mutation model with equal forward
            and back mutation rates theta
    """
    nu1,nu2,T1,T2 = params
    y = equilibrium(rho, theta, ism=ism, order=order)
    y = Numerics.integrate(y, T1, rho=rho, theta=theta, nu=nu1, order=order, dt=0.001, ism=ism)
    y = Numerics.integrate(y, T2, rho=rho, theta=theta, nu=nu2, order=order, dt=0.001, ism=ism)
    y = LDstats(y, num_pops=1, order=order)
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_onepop(y, n=ns, order=order)
        else:
            return Corrections.corrected_onepop_genotypes(y, n=ns/2, order=order)
    else:
        return y

def growth(params, order=2, rho=0, theta=0.0008, ns=200, corrected=False, ism=False):
    """
    Exponential growth (or decay) model
    params: = (nuF,T), nu F is the final population size after time T (starting from nu=1)
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for sampling bias in sample
               size ns
    ism: if True, we assume an infinite site model
         if False (default), we assume a reversible mutation model with equal forward
            and back mutation rates theta
    """
    nuF,T = params
    y = equilibrium(rho, theta, ism=ism, order=order)
    nu_func = lambda t: np.exp( np.log(nuF) *t/T)
    y = Numerics.integrate(y, T, rho=rho, theta=theta, nu=nu_func, order=order, dt=0.001, ism=ism)
    y = LDstats(y, num_pops=1, order=order)
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_onepop(y, n=ns, order=order)
        else:
            return Corrections.corrected_onepop_genotypes(y, n=ns/2, order=order)
    else:
        return y

def bottlegrowth(params, ns=200, rho=0, theta=0.0008, order=2, corrected=False, ism=False, genotypes=False):    
    """
    Exponential growth (or decay) model after size change
    params: = (nuB,nuF,T), nu F is the final population size after time T, 
                starting from instantaneous population size change of nuB
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for sampling bias in sample
               size ns
    ism: if True, we assume an infinite site model
         if False (default), we assume a reversible mutation model with equal forward
            and back mutation rates theta
    """
    nuB,nuF,T = params
    y = equilibrium(rho, theta, ism=ism, order=order)
    nu_func = lambda t: nuB * np.exp( np.log(nuF/nuB) *t/T)
    y = Numerics.integrate(y, T, rho=rho, theta=theta, nu=nu_func, order=order, dt=0.001, ism=ism)
    y = LDstats(y, num_pops=1, order=order)
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_onepop(y, n=ns, order=order)
        else:
            return Corrections.corrected_onepop_genotypes(y, n=ns/2, order=order)
    else:
        return y

