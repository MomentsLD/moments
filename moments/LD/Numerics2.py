import numpy as np

import copy
import itertools
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as spinv
import pickle

import Matrices2
import Util

"""
splits, marginalizations, and other manipulations
"""

def split_h(h, pop_to_split, num_pops):        
    h_from = h
    h_new = np.empty(int((num_pops+1)*(num_pops+2)/2))
    c = 0
    hn = Util.het_names(num_pops)
    for ii in range(1,num_pops+2):
        for jj in range(ii,num_pops+2):
            if jj == num_pops+1:
                if ii == jj:
                    h_new[c] = h_from[hn.index('H_{0}_{0}'.format(pop_to_split))]
                else:
                    if ii <= pop_to_split:
                        h_new[c] = h_from[hn.index('H_{0}_{1}'.format(ii,pop_to_split))]
                    else:
                        h_new[c] = h_from[hn.index('H_{0}_{1}'.format(pop_to_split,ii))]
            else:
                h_new[c] = h_from[hn.index('H_{0}_{1}'.format(ii,jj))]
            
            c += 1
    
    return h_new

def split_ld(y, pop_to_split, num_pops):
    mom_list_from = Util.ld_names(num_pops)
    mom_list_to = Util.ld_names(num_pops+1)
    
    y_new = np.ones(len(mom_list_to))
    for ii,mom_to in enumerate(mom_list_to):
        if mom_to in mom_list_from:
            y_new[ii] = y[mom_list_from.index(mom_to)]
        else:
            mom_to_split = mom_to.split('_')
            for jj in range(1, len(mom_to_split)):
                if int(mom_to_split[jj]) == num_pops+1:
                    mom_to_split[jj] = str(pop_to_split)
            mom_from = '_'.join(mom_to_split)
            y_new[ii] = y[mom_list_from.index(Util.map_moment(mom_from))]
    return y_new
                
                
def marginalize_h(h, p):
    pass

def marginalize_ld(y, p):
    pass

### transition matrices

def drift(num_pops, nus):
    Dh = Matrices2.drift_h(num_pops, nus)
    Dld = Matrices2.drift_ld(num_pops, nus)
    return Dh, Dld

def mutation(num_pops, theta, frozen=None):
    ### mutation for ld also has dependence on H
    Uh = Matrices2.mutation_h(num_pops, theta, frozen=frozen)
    Uld = Matrices2.mutation_ld(num_pops, theta, frozen=frozen)
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

def integrate(Y, nu, T, dt=0.001, theta=0.001, rho=None, m=None, num_pops=None, frozen=None):
    """
    
    """
    if num_pops == None:
        num_pops = len(nu)
    
    h = Y[-1]
    if len(Y) == 2:
        y = Y[0]
    else:
        ys = Y[:-1]
    
    if callable(nu):
        nus = nu(0)
    else:
        nus = [np.float(nu_pop) for nu_pop in nu]
    
    Uh, Uld = mutation(num_pops, theta, frozen=frozen)
    
    if rho is not None:
        # if rho is a scalar, return single matrix, if rho is a list, returns list of matrices
        R = recombination(num_pops, rho=rho)
    
    if num_pops > 1 and m is not None:
        Mh, Mld = migration(num_pops, m)
    
    
    dt_last = dt
    nus_last = nus
    elapsed_t = 0
    
    while elapsed_t < T:
        if elapsed_t + dt > T:
            dt = T-elapsed_t
        
        if callable(nu):
            nus = nu(elapsed_t+dt/2.)
        
        if dt != dt_last or nus != nus_last or elapsed_t == 0:
            Dh, Dld = drift(num_pops, nus)
            # check if we need migration matrics
            if num_pops > 1 and m is not None: # with migration
                Ab_h = Dh+Mh
                # check if we need LD matrices, and then if we need one or many
                if rho is not None:
                    if np.isscalar(rho):
                        Ab_ld = Dld+Mld+R
                    else:
                        Ab_ld = [Dld+Mld+R[i] for i in range(len(rho))]
            else: # no migration
                Ab_h = Dh
                # check if we need LD matrices, and then if we need one or many
                if rho is not None:
                    if np.isscalar(rho):
                        Ab_ld = Dld+R
                    else:
                        Ab_ld = [Dld+R[i] for i in range(len(rho))]
            
            # heterozygosity solver
            Ab1_h = np.eye(Ab_h.shape[0]) + dt/2.*Ab_h
            Ab2_h = np.linalg.inv(np.eye(Ab_h.shape[0]) - dt/2.*Ab_h)
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
                y = Ab1_ld.dot(y) + dt*Uld.dot(h)
            else:
                ys = [Ab1_ld[i].dot(ys[i]) +  + dt*Uld.dot(h) for i in range(len(ys))]
        # h
        h = Ab1_h.dot(h) + dt*Uh
        
        # backward
        #ld
        if rho is not None:
            if np.isscalar(rho):
                y = Ab2_ld(y)
            else:
                ys = [Ab2_ld[i](ys[i]) for i in range(len(ys))]
        #h
        h = Ab2_h.dot(h)

        elapsed_t += dt
        dt_last = copy.copy(dt)
        nus_last = copy.copy(nus)

    
    Y[-1] = h
    if np.isscalar(rho):
        Y[0] = y
    else:
        Y[:-1] = ys[:-1]
        
    return Y

def steady_state(theta=0.001, rho=None):
    h_ss = [theta]
    if hasattr(rho, '__len__'): # list of rhos
        ys_ss = [equilibrium_ld(theta=theta, rho=r) for r in rho]
        return ys_ss + [h_ss]
    elif np.isscalar(rho): # one rho value
        y_ss = equilibrium_ld(theta=theta, rho=rho)
        return [y_ss, h_ss]
    else: # only het stats
        return [h_ss]

def equilibrium_ld(theta=0.001, rho=0.0):
    h_ss = [theta]
    U = Matrices2.mutation_ld(1, theta)
    R = Matrices2.recombination(1, rho)
    D = Matrices2.drift_ld(1, [1.])
    return factorized(D+R)(-U.dot(h_ss))


