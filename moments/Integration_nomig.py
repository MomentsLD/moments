import numpy as np
import scipy as sp
import time
from scipy.sparse import linalg

import Spectrum_mod
import Jackknife as jk
import LinearSystem_1D as ls1
import Tridiag_solve as ts
#------------------------------------------------------------------------------
# Functions for the computation of the Phi-moments for multidimensional models:
# we integrate the ode system on the Phi_n(i) to compute their evolution
# we write it (and solve it) as an approximated linear system:
#       Phi_n' = Bn(N) + (1/(4N)Dn + S1n + S2n)Phi_n
# where :
#       N is the total population size
#       Bn(N) is the mutation source term
#       1/(4N)Dn is the drift effect matrix
#       S1n is the selection matrix for h = 0.5
#       S2n is the effect of h != 0.5
#------------------------------------------------------------------------------

#-----------------------------------
# functions to compute the matrices-
#-----------------------------------

# Mutations
def calcB(dims, u):
    B = np.zeros(dims)
    for k in range(len(dims)):
        ind = np.zeros(len(dims), dtype = 'int')
        ind[k] = int(1)
        tp = tuple(ind)
        B[tp] = dims[k] - 1
    return u * B

#----------------------------------
# updates for the time integration-
#----------------------------------

# 2D
def ud1_2pop_1(sfs, Q):
    for i in range(int(sfs.shape[1])):
        sfs[:,i] = Q[0].dot(sfs[:,i])
    return sfs

def ud1_2pop_2(sfs, Q):
    for i in range(int(sfs.shape[0])):
        sfs[i,:] = Q[1].dot(sfs[i,:])
    return sfs

def ud2_2pop_1(sfs, slv):
    for i in range(int(sfs.shape[1])):
        sfs[:,i] = slv[0](sfs[:,i])
    return sfs

def ud2_2pop_2(sfs, slv):
    for i in range(int(sfs.shape[0])):
        sfs[i,:] = slv[1](sfs[i,:])
    return sfs

# for 3D, 4D and 5D cases, each couple of directions are coded separately to simplify the permutations...
#------------------------------
# 3D
# step 1

def ud1_3pop_1(sfs, Q):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[1])):
            sfs[:, j, i] = Q[0].dot(sfs[:, j, i])
    return sfs

def ud1_3pop_2(sfs, Q):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[0])):
            sfs[j, :, i] = Q[1].dot(sfs[j, :, i])
    return sfs

def ud1_3pop_3(sfs, Q):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[0])):
            sfs[j, i, :] = Q[2].dot(sfs[j, i, :])
    return sfs

# step 2
def ud2_3pop_1(sfs, slv):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[1])):
            sfs[:, j, i] = slv[0](sfs[:, j, i])
    return sfs

def ud2_3pop_2(sfs, slv):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[0])):
            sfs[j, :, i] = slv[1](sfs[j, :, i])
    return sfs

def ud2_3pop_3(sfs, slv):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[0])):
            sfs[j, i, :] = slv[2](sfs[j, i, :])
    return sfs

#------------------------------
# 4D
# step 1
def ud1_4pop_1(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                sfs[i, j, k, :] = Q[3].dot(sfs[i, j, k, :])
    return sfs

def ud1_4pop_2(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                sfs[i, j, :, k] = Q[2].dot(sfs[i, j, :, k])
    return sfs

def ud1_4pop_3(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[i, :, j, k] = Q[1].dot(sfs[i, :, j, k])
    return sfs

def ud1_4pop_4(sfs, Q):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[:, i, j, k] = Q[0].dot(sfs[:, i, j, k])
    return sfs

# step 2
def ud2_4pop_1(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                sfs[i, j, k, :] = slv[3](sfs[i, j, k, :])
    return sfs

def ud2_4pop_2(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                sfs[i, j, :, k] = slv[2](sfs[i, j, :, k])
    return sfs

def ud2_4pop_3(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[i, :, j, k] = slv[1](sfs[i, :, j, k])
    return sfs

def ud2_4pop_4(sfs, slv):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[:, i, j, k] = slv[0](sfs[:, i, j, k])
    return sfs


#------------------------------
# 5D
# step 1
def ud1_5pop_1(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[3])):
                    sfs[i, j, k, l, :] = Q[4].dot(sfs[i, j, k, l, :])
    return sfs

def ud1_5pop_2(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, k, :, l] = Q[3].dot(sfs[i, j, k, :, l])
    return sfs

def ud1_5pop_3(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, :, k, l] = Q[2].dot(sfs[i, j, :, k, l])
    return sfs

def ud1_5pop_4(sfs, Q):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, :, j, k, l] = Q[1].dot(sfs[i, :, j, k, l])
    return sfs

def ud1_5pop_5(sfs, Q):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[:, i, j, k, l] = Q[0].dot(sfs[:, i, j, k, l])
    return sfs

# step 2
def ud2_5pop_1(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[3])):
                    sfs[i, j, k, l, :] = slv[4](sfs[i, j, k, l, :])
    return sfs

def ud2_5pop_2(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, k, :, l] = slv[3](sfs[i, j, k, :, l])
    return sfs

def ud2_5pop_3(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, :, k, l] = slv[2](sfs[i, j, :, k, l])
    return sfs

def ud2_5pop_4(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, :, j, k, l] = slv[1](sfs[i, :, j, k, l])
    return sfs

def ud2_5pop_5(sfs, slv):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[:, i, j, k, l] = slv[0](sfs[:, i, j, k, l])
    return sfs

# neutral case step 2 (tridiag solver)
# 2D
def udn2_2pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        sfs[:,i] = ts.solve(A[0], Di[0], C[0], sfs[:,i])
    return sfs

def udn2_2pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        sfs[i,:] = ts.solve(A[1], Di[1], C[1], sfs[i,:])
    return sfs

# 3D
def udn2_3pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[1])):
            sfs[:, j, i] = ts.solve(A[0], Di[0], C[0], sfs[:, j, i])
    return sfs

def udn2_3pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[0])):
            sfs[j, :, i] = ts.solve(A[1], Di[1], C[1], sfs[j, :, i])
    return sfs

def udn2_3pop_3(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[0])):
            sfs[j, i, :] = ts.solve(A[2], Di[2], C[2], sfs[j, i, :])
    return sfs

# 4D
def udn2_4pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                sfs[i, j, k, :] = ts.solve(A[3], Di[3], C[3], sfs[i, j, k, :])
    return sfs

def udn2_4pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                sfs[i, j, :, k] = ts.solve(A[2], Di[2], C[2], sfs[i, j, :, k])
    return sfs

def udn2_4pop_3(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[i, :, j, k] = ts.solve(A[1], Di[1], C[1], sfs[i, :, j, k])
    return sfs

def udn2_4pop_4(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[:, i, j, k] = ts.solve(A[0], Di[0], C[0], sfs[:, i, j, k])
    return sfs

# 5D
def udn2_5pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[3])):
                    sfs[i, j, k, l, :] = ts.solve(A[4], Di[4], C[4], sfs[i, j, k, l, :])
    return sfs

def udn2_5pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, k, :, l] = ts.solve(A[3], Di[3], C[3], sfs[i, j, k, :, l])
    return sfs

def udn2_5pop_3(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, :, k, l] = ts.solve(A[2], Di[2], C[2], sfs[i, j, :, k, l])
    return sfs

def udn2_5pop_4(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, :, j, k, l] = ts.solve(A[1], Di[1], C[1], sfs[i, :, j, k, l])
    return sfs

def udn2_5pop_5(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[:, i, j, k, l] = ts.solve(A[0], Di[0], C[0], sfs[:, i, j, k, l])
    return sfs

# sfs update 
def update_step1(sfs, Q):
    assert(len(Q) == len(sfs.shape))
    for i in range(len(sfs.shape)):
        sfs = eval('ud1_'+str(len(sfs.shape))+'pop_'+str(i+1)+'(sfs, Q)')
    return sfs

def update_step2(sfs, slv):
    assert(len(slv) == len(sfs.shape))
    for i in range(len(sfs.shape)):
        sfs = eval('ud2_'+str(len(sfs.shape))+'pop_'+str(i+1)+'(sfs, slv)')
    return sfs

def update_step2_neutral(sfs, A, Di, C):
    assert(len(A) == len(sfs.shape))
    for i in range(len(sfs.shape)):
        sfs = eval('udn2_'+str(len(sfs.shape))+'pop_'+str(i+1)+'(sfs, A, Di, C)')
    return sfs

#--------------------
# Integration in time
#--------------------
# N : total population size (vector N = (N1,...,Np))
# n : samples size (vector n = (n1,...,np))
# tf : final simulation time (/2N1 generations)
# gamma : selection coefficients (vector gamma = (gamma1,...,gammap))
# theta : mutation rate
# h : allele dominance (vector h = (h1,...,hp))
# m : migration rates matrix (2D array, m[i,j] is the migration rate from pop j to pop i, normalized by 1/4N1)

# for a "lambda" definition of N - with backward Euler integration scheme
# where t is the relative time in generations such as t = 0 initially
# Npop is a lambda function of the time t returning the vector N = (N1,...,Np) or directly the vector if N does not evolve in time

def integrate_nomig(sfs0, Npop, n, tf, dt_fac = 0.05, gamma = None, h = None, theta = 1.0):
    # neutral case if the parameters are not provided
    if gamma is None : gamma = np.zeros(len(n))
    if h is None : h = 0.5 * np.ones(len(n))
    
    sfs0 = np.array(sfs0)
    # parameters of the equation
    if callable(Npop): N = np.array(Npop(0))
    else : N = np.array(Npop)
    
    Nold = N + np.ones(len(N))
    s = np.array(gamma)
    h = np.array(h)
    Tmax = tf * 2.0
    dt = Tmax * dt_fac
    u = theta / 4.0
    # dimensions of the sfs
    dims = n + np.ones(len(n))
    d = int(np.prod(dims))

    # we compute the matrices we will need
    ljk = [jk.calcJK13(int(dims[i]-1)) for i in range(len(dims))]
    ljk2 = [jk.calcJK23(int(dims[i]-1)) for i in range(len(dims))]
    
    # drift
    vd = [ls1.calcD(np.array(dims[i])) for i in range(len(dims))]
    
    # selection part 1
    vs = [ls1.calcS(dims[i], ljk[i]) for i in range(len(n))]
    S1 = [s[i] * h[i] * vs[i] for i in range(len(n))]
    
    # selection part 2
    vs2 = [ls1.calcS2(dims[i], ljk2[i]) for i in range(len(n))]
    S2 = [s[i] * (1-2.0*h[i]) * vs[i] for i in range(len(n))]
    
    # mutations
    B = calcB(dims, u)

    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        dt_old = dt
        if t+dt > Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold != N).any() or dt != dt_old:
            D = [1.0 / 4 / N[i] * vd[i] for i in range(len(dims))]
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc') - dt/2.0*(D[i]+S1[i]+S2[i])) for i in range(len(n))]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc') + dt/2.0*(D[i]+S1[i]+S2[i]) for i in range(len(n))]
        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs + dt*B)
        elif len(n) > 1:
            sfs = update_step1(sfs, Q)
            sfs = update_step2(sfs + dt*B, slv)
        Nold = N
        t += dt
        # we update the value of N if a function was provided as argument
        if callable(Npop) : N = np.array(Npop(t / 2.0))
    return Spectrum_mod.Spectrum(sfs)

def integrate_neutral(sfs0, Npop, n, tf, dt_fac = 0.05, theta = 1.0):
    sfs0 = np.array(sfs0)
    # parameters of the equation
    if callable(Npop): N = np.array(Npop(0))
    else : N = np.array(Npop)
    
    Nold = N + np.ones(len(N))
    Tmax = tf * 2.0
    dt = Tmax * dt_fac
    u = theta / 4.0
    # dimensions of the sfs
    dims = n + np.ones(len(n))
    d = int(np.prod(dims))
    
    # drift
    vd = [ls1.calcD_dense(dims[i]) for i in range(len(n))]
    diags = [ts.mat_to_diag(x) for x in vd]

    # mutations
    B = calcB(dims, u)

    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        dt_old = dt
        if t+dt > Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold != N).any() or dt != dt_old:
            D = [1.0 / 4 / N[i] * vd[i] for i in range(len(n))]
            A = [-0.5 * dt/ 4 / N[i] * diags[i][0] for i in range(len(n))]
            Di = [np.ones(dims[i])-0.5 * dt / 4 / N[i] * diags[i][1] for i in range(len(n))]
            C = [-0.5 * dt/ 4 / N[i] * diags[i][2] for i in range(len(n))]
            # system inversion for backward scheme
            for i in range(len(n)):
                ts.factor(A[i], Di[i], C[i])
            Q = [np.eye(dims[i]) + 0.5*dt*D[i] for i in range(len(n))]
        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = ts.solve(A[0], Di[0], C[0], np.dot(Q[0], sfs) + dt*B)
        else:
            sfs = update_step1(sfs, Q)
            sfs = update_step2_neutral(sfs + dt*B, A, Di, C)
        Nold = N
        t += dt
        # we update the value of N if a function was provided as argument
        if callable(Npop) : N = np.array(Npop(t / 2.0))
    return Spectrum_mod.Spectrum(sfs)
