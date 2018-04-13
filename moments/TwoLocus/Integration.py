import numpy as np
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
import moments.TwoLocus.Jackknife
import moments.TwoLocus.Numerics

import warnings
warnings.filterwarnings("ignore")

### XXX 4/11/18 - for finite genome model, need to make sure density is conserved, track density moving to fixed states, etc, check along surface, and so on

def integrate(F, nu, tf, rho=0.0, dt=0.01, theta=1.0, gamma=0.0, h=0.5, sel_params=None,
              finite_genome=False, u=None, v=None):
    """
    There are two selection options:
    1) set gamma and h, which is for selection at the left locus, linked to a neutral locus
    2) set sel_params, which are the (additive) selection coefficients for haplotypes AB, Ab, and aB
        so that sel_params = (sAB, sA, sB)
        If sAB = sA + sB, no epistatic interaction
    """
    if tf <= 0:
        return F
    
    n = len(F) - 1
    
    if callable(nu):
        N = nu(0)
    else:
        N = nu
    
    N_old = 1.0
    
    if finite_genome is False:
        M_0to1, M = moments.TwoLocus.Numerics.mutations(n, theta=theta)
        D = moments.TwoLocus.Numerics.drift(n)
    elif finite_genome is True:
        if u is None or v is None:
            raise ValueError("if finite genome, must specify u and v")
        M = moments.TwoLocus.Numerics.mutations_reversible(n, u, v)
        D = moments.TwoLocus.Numerics.drift_reversible(n)

    Phi = moments.TwoLocus.Numerics.array_to_Phi(F)
    
    t_elapsed = 0
    
    if rho == 0:
        # check which selection regime is used
        if sel_params is not None:
            J1 = moments.TwoLocus.Jackknife.calc_jk(n,1)
            S = moments.TwoLocus.Numerics.selection_two_locus(n, sel_params)
            while t_elapsed < tf:
                dt_old = dt
                if t_elapsed + dt > tf:
                    dt = tf-t_elapsed
                
                if callable(nu):
                    N = nu(t_elapsed + dt/2.)
                
                if t_elapsed == 0 or N_old != N or dt != dt_old:
                    Ab = M/2. + S.dot(J1) + D/(2.*N)
                    Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                    slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                
                if finite_genome is False:
                    Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                else:
                    Phi = slv(Ab1.dot(Phi))
                
                N_old = N
                t_elapsed += dt
        elif gamma == 0:
            while t_elapsed < tf:
                dt_old = dt
                if t_elapsed + dt > tf:
                    dt = tf-t_elapsed
                
                if callable(nu):
                    N = nu(t_elapsed + dt/2.)
                
                if t_elapsed == 0 or N_old != N or dt != dt_old:
                    Ab = M/2. + D/(2.*N)
                    Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                    slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                
                if finite_genome is False:
                    Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                else:
                    Phi = slv(Ab1.dot(Phi))
                
                N_old = N
                t_elapsed += dt
        elif gamma != 0:
            Sa = 2*gamma*h * moments.TwoLocus.Numerics.selection_additive_component(n)
            J1 = moments.TwoLocus.Jackknife.calc_jk(n,1)
            if h == 0.5:
                while t_elapsed < tf:
                    dt_old = dt
                    if t_elapsed + dt > tf:
                        dt = tf-t_elapsed
                    
                    if callable(nu):
                        N = nu(t_elapsed + dt/2.)
                    
                    if t_elapsed == 0 or N_old != N or dt != dt_old:
                        Ab = M/2. + Sa.dot(J1) + D/(2.*N)
                        Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                        slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                    
                    if finite_genome is False:
                        Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                    else:
                        Phi = slv(Ab1.dot(Phi))
                    
                    N_old = N
                    t_elapsed += dt
            else:
                Sd = 2*gamma*(1-2*h) * moments.TwoLocus.Numerics.selection_dominance_component(n)
                J2 = moments.TwoLocus.Jackknife.calc_jk(n,2)
                while t_elapsed < tf:
                    dt_old = dt
                    if t_elapsed + dt > tf:
                        dt = tf-t_elapsed
                    
                    if callable(nu):
                        N = nu(t_elapsed + dt/2.)
                    
                    if t_elapsed == 0 or N_old != N or dt != dt_old:
                        Ab = M/2. + Sa.dot(J1) + Sd.dot(J2) + D/(2.*N)
                        Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                        slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                    
                    if finite_genome is False:
                        Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                    else:
                        Phi = slv(Ab1.dot(Phi))
                    
                    N_old = N
                    t_elapsed += dt
    elif rho > 0:
        J1 = moments.TwoLocus.Jackknife.calc_jk(n,1)
        if finite_genome is False:
            R = moments.TwoLocus.Numerics.recombination(n,rho)
        else:
            R = moments.TwoLocus.Numerics.recombination_reversible(n,rho)
        if sel_params is not None:
            S = moments.TwoLocus.Numerics.selection_two_locus(n, sel_params)
            while t_elapsed < tf:
                dt_old = dt
                if t_elapsed + dt > tf:
                    dt = tf-t_elapsed
                
                if callable(nu):
                    N = nu(t_elapsed + dt/2.)
                
                if t_elapsed == 0 or N_old != N or dt != dt_old:
                    Ab = M/2. + R.dot(J1) + S.dot(J1) + D/(2.*N)
                    Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                    slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                
                if finite_genome is False:
                    Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                else:
                    Phi = slv(Ab1.dot(Phi))
                
                N_old = N
                t_elapsed += dt

        elif gamma == 0:
            while t_elapsed < tf:
                dt_old = dt
                if t_elapsed + dt > tf:
                    dt = tf-t_elapsed
                
                if callable(nu):
                    N = nu(t_elapsed + dt/2.)
                
                if t_elapsed == 0 or N_old != N or dt != dt_old:
                    Ab = D/(2.*N) + R.dot(J1) + M/2.
                    Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                    slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                
                if finite_genome is False:
                    Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                else:
                    Phi = slv(Ab1.dot(Phi))
                
                N_old = N
                t_elapsed += dt
        else:
            Sa = 2*gamma*h * moments.TwoLocus.Numerics.selection_additive_component(n)
            if h == 0.5:
                while t_elapsed < tf:
                    dt_old = dt
                    if t_elapsed + dt > tf:
                        dt = tf-t_elapsed
                    
                    if callable(nu):
                        N = nu(t_elapsed + dt/2.)
                    
                    if t_elapsed == 0 or N_old != N or dt != dt_old:
                        Ab = D/(2.*N) + R.dot(J1) + Sa.dot(J1) + M/2.
                        Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                        slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                    
                    if finite_genome is False:
                        Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                    else:
                        Phi = slv(Ab1.dot(Phi))
                    
                    N_old = N
                    t_elapsed += dt
            else:
                Sd = 2*gamma*(1-2*h) * moments.TwoLocus.Numerics.selection_dominance_component(n)
                J2 = moments.TwoLocus.Jackknife.calc_jk(n,2)
                while t_elapsed < tf:
                    dt_old = dt
                    if t_elapsed + dt > tf:
                        dt = tf-t_elapsed
                    
                    if callable(nu):
                        N = nu(t_elapsed + dt/2.)
                    
                    if t_elapsed == 0 or N_old != N or dt != dt_old:
                        Ab = D/(2.*N) + R.dot(J1) + Sa.dot(J1) + Sd.dot(J2) + M/2.
                        Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                        slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                    
                    if finite_genome is False:
                        Phi = slv(Ab1.dot(Phi) + dt*M_0to1)
                    else:
                        Phi = slv(Ab1.dot(Phi))
                    
                    N_old = N
                    t_elapsed += dt
    else:
        print "rho must be >= 0"
    return moments.TwoLocus.Numerics.Phi_to_array(Phi,n)

        
