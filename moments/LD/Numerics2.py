import numpy as np

import copy
import itertools
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as spinv
import pickle

from . import Matrices2

"""
Heterozygosity and LD stat names
"""

### handling data (splits, marginalization, admixture, ...)
def moment_names(num_pops):
    """
    num_pops : number of populations, indexed [1,...,num_pops]
    """
    pass

"""
We need to map moments for split and rearrangement functions
"""
mom_map = {}
def map_moment(mom)
    """
    
    """
    pass

def split_h(h, p):
    pass

def split_ld(y, p):
    pass

def marginalize_h(h, p):
    pass

def marginalize_ld(y, p):
    pass

### transition matrices

def drift(num_pops, nus):
    Dh = Matrices2.drift_h(num_pops, nus)
    Dld = Matrices2.drift_ld(num_pops, nus)
    return Dh, Dld

def mutation(num_pops, theta):
    ### mutation for ld also has dependence on H
    Uh = Matrices2.mutation_h(num_pops, theta)
    Uld = Matrices2.mutation_ld(num_pops, theta)
    return Uh, Uld

def recombination(num_pops, rho=0.0):
    if np.isscalar(rho):
        R = Matrices2.recombination(num_pops, rho)
    else:
        R = [Matrices2.recombination(num_pops, r) for r in rho]
    return R

def migration(num_pops, m):
    Mh = Matrices2.migration_h(num_pops, m)
    Mld = Matrices2.migration_ld(num_pops, m)
    return Mh, Mld


### integration routines

def integrate(Y, nu, T, dt=0.001, theta=0.001, rho=None, m=None, num_pops=None):
    """
    
    """
    if num_pops == None:
        raise num_pops = len(nu)
    
    h = Y[-1]
    if len(Y) == 2:
        y = Y[0]
    else:
        ys = Y[:-1]
    
    if callable(nu):
        nus = nu(0)
    else:
        nus = [np.float(nu_pop) for nu_pop in nu]
    
    Uh, Uld = mutation(num_pops, theta)
    
    if rho is not None:
        # if rho is a scalar, return single matrix, if rho is a list, returns list of matrices
        R = recombination(num_pops, rho=rho)
    
    if num_pops > 1 and m is not None:
        Mh, Mld = migration(num_pops, m)
    
    
    dt_last = dt
    nus_last = nus
    elapsed_t = 0
    
    while elapsed_t < T:
        if elapsed_t + dt > T
            dt = T-elapsed_t
        
        if callable(nu):
            nus = nu(elapsed_t+dt/2.)
        
        if dt != dt_last or nus != nus_last or elapsed_t == 0:
            Dh, Dld = drift(num_pops, nus)
            # check if we need migration matrics
            if num_pops > 1 and m is not None: # with migration
                Ab_h = Dh+Mh+Uh
                # check if we need LD matrices, and then if we need one or many
                if rho is not None:
                    if np.isscalar(rho):
                        Ab_ld = Dld+Mld+R+Uld
                    else:
                        Ab_ld = [Dld+Mld+R[i]+Uld for i in range(len(rho))]
            else: # no migration
                Ab_h = Dh+Uh
                # check if we need LD matrices, and then if we need one or many
                if rho is not None:
                    if np.isscalar(rho):
                        Ab_ld = Dld+R+Uld
                    else:
                        Ab_ld = [Dld+R[i]+Uld for i in range(len(rho))]
            
            # heterozygosity solver
            Ab1_h = identity(Ab_h.shape[0], format='csc') + dt/2.*Ab_h
            Ab2_h = factorized(identity(Ab_h.shape[0], format='csc') - dt/2.*Ab_h)
            # ld solvers
            ### also need to have the mutation(ld) depending on H!!!
            if rho is not None:
                if np.isscalar(rho):
                    Ab1_ld = identity(Ab_ld.shape[0], format='csc') + dt/2.*Ab_ld
                    Ab2_ld = factorized(identity(Ab_ld.shape[0], format='csc') - dt/2.*Ab_ld)
                else:
                    Ab1_ld = [identity(Ab_ld[i].shape[0], format='csc') + dt/2.*Ab_ld[i] for i in range(len(rho))]
                    Ab2_ld = [factorized(identity(Ab_ld[i].shape[0], format='csc') - dt/2.*Ab_ld[i]) for i in range(len(rho))]
        
        # forward
        # ld
        if rho is not None:
            if np.isscalar(rho):
                y = Ab1_ld.dot(ys)
            else:
                ys = [Ab1_ld[i].dot(ys[i]) + xxx_mutation_dep_on_h for i in range(len(ys))]
        # h
        h = Ab1_h.dot(h)
        
        # backward
        #ld
        if rho is not None:
            if np.isscalar(rho):
                y = Ab2_ld(y) # how can we make this depend on h...
            else:
                ys = [Ab2_ld[i](ys[i]) for i in range(len(ys))]
        #h
        h = Ab2_h(h)
    
    Y[-1] = h
    if np.isscalar(rho):
        Y[0] = y
    else:
        Y[:-1] = ys[:-1]
        
    return Y

def steady_state(theta=0.001, rho=None):
    if rho == None: # only het stats
    
    elif np.isscalar(rho): # one rho value
    
    else: # list of rhos

