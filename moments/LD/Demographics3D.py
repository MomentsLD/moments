import numpy as np
from moments.LD import Numerics
from moments.LD import Corrections
from moments.LD.LDstats_mod import LDstats

def admix(params, order=2, rho=0, theta=0.0008, ns=[20,20,20], 
          corrected=False, genotypes=False):
    """
    Admixture model:
        |
        |
       / \   
      /   \  T1
     /\___/\  <-- admixture event between two populations that split
    /   |   \  T2
    A   C   B
    
    A and B are the two parental populations, C (is third position) 
    
    params: (nu1,nu2,nu3,T1,T2,f)
            f = proportion of admixture from A (1-f from B)
            T1 = time between split of A and B to admixture event
            T2 = time between admixture event and present
            nu1,nu2,nu3 = sizes of A, B, and C
    order: order of D statistics (e.g. order=2 gives the D^2 system)
    rho: population-scaled recombination rate (4Nr)
    theta: population-scaled mutation rate (4Nu)
    ns: haploid sample size, used if corrected=True
    corrected: if True, returns statistics that accounts for 
               sampling bias in sample size ns    
    Standard neutral model, with populations never diverging
    """
    if order != 2:
        raise ValueError("We can only run multipopulation demographies for order 2 statistics.")
    
    nu1,nu2,nu3,f,T1,T2 = params
    
    n1,n2,n3 = ns
    
    y = Numerics.equilibrium_multipop(rho, theta, ism=ism)
    y = LDstats(y, num_pops=1, order=order)
    
    y = y.split(1)
    y.integrate([nu1,nu2], T1, rho=rho, theta=theta, ism=ism)
    y = y.admix(f)
    y.integrate([nu1,nu2,nu3], T2, rho=rho, theta=theta, ism=ism)
    
    if corrected == True:
        if genotypes == False:
            return Corrections.corrected_multipop(y, ns=(n1,n2, n3), num_pops=3)
        else:
            return Corrections.corrected_multipop_genotypes(y, ns=(n1/2,n2/2,n3/2), num_pops=3)
    else:
        return y


