import numpy as np, math
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from scipy.sparse import identity
import moments.Triallele.Numerics
import moments.Triallele.Jackknife

"""
Integration for triallele model
We use a Crank-Nicolson scheme to integrate the fs forward in time:

    Numerics to solve the ODE
    d/dt Phi_n = 1/2 D Phi_n + mutation
    D - drift operator, scaled by 2N
    B - infinite sites model

    We don't track density along the diagonal axis (where the ancestral allele was lost)
    The horizontal and vertical axes track the background biallelic frequency spectra,
        and is unaffected by the interior triallelic spectrum
    Mutation introduces density from background biallelic spectrum, but doesn't remove density

    size of Phi is (n+1)(n+2)/2

"""

def integrate_cn(F, nu, tf, dt=0.01, gammas=None, theta=1.0):
    if tf <= 0:
        print('Integration time should be positive.')
        return F
    
    ns = len(F) - 1
    
    if callable(nu):
        N = nu(0)
    else:
        N = nu
    
    N_old = 1.0
    
    D = moments.Triallele.Numerics.drift(ns)
    B_bi, B_tri = moments.Triallele.Numerics.mutation(ns, theta, theta, theta)
    
    Phi = moments.Triallele.Numerics.flatten(F)
    
    if gammas is None:
        gammas = (0,0,0,0,0)
    
    if np.any(gammas) == False:
        t_elapsed = 0
        while t_elapsed < tf:
            dt_old = dt
            if t_elapsed + dt > tf:
                dt = tf - t_elapsed
            
            if callable(nu):
                N = nu(t_elapsed + dt/2.)
            
            if t_elapsed == 0 or N_old != N or dt != dt_old:
                Ab = B_tri+D/(2.*N)
                Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
            
            Phi = slv(Ab1.dot(Phi)+dt*B_bi)
            
            N_old = N
            t_elapsed += dt
        return moments.Triallele.Numerics.reform(Phi,ns)
    else:
        S = moments.Triallele.Numerics.selection(ns, gammas)
        J = moments.Triallele.Jackknife.calcJK_2(ns)
        
        t_elapsed = 0
        while t_elapsed < tf:
            dt_old = dt
            if t_elapsed + dt > tf:
                dt = tf - t_elapsed
            
            if callable(nu):
                N = nu(t_elapsed + dt/2.)
            
            if t_elapsed == 0 or N_old != N or dt != dt_old:
                Ab = D/(2.*N) + S.dot(J) + B_tri
                #Ab1 = identity(Ab.shape[0]) + dt/2.*Ab
                #slv = factorized(identity(Ab.shape[0]) - dt/2.*Ab)
                Ab_fd = identity(Ab.shape[0]) + dt*Ab
            
            Phi = Ab_fd.dot(Phi) + dt*B_bi
            #Phi = slv(Ab1.dot(Phi)+dt*B_bi)
            
            N_old = N
            t_elapsed += dt
        return moments.Triallele.Numerics.reform(Phi,ns)
    
