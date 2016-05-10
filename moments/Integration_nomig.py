import numpy as np
import scipy as sp
import time
from scipy.sparse import linalg

import Spectrum_mod
import Jackknife as jk
import LinearSystem_1D as ls1
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

def to_good_shape_banded(mat, l, u):
    res = np.zeros([l+u+1, mat.shape[1]])
    for i in range(mat.shape[0]):
        for j in range(max(0, i-l), min(mat.shape[1], i+u+1)):
            res[u+i-j, j] = mat[i, j]
    return res

def id_banded(d, l, u):
    res = np.zeros([l+u+1, d])
    for i in range(int(d)):
        res[u, i] = 1.0
    return res


# Mutations
def calcB(dims, u):
    B = np.zeros(dims)
    for k in range(len(dims)):
        ind = np.zeros(len(dims), dtype = 'int')
        ind[k] = int(1)
        tp = tuple(ind)
        B[tp] = dims[k] - 1
    return u * B

# Drift
def calcD(dims):
    res = []
    for i in range(len(dims)):
        res.append(ls1.calcD(np.array(dims[i])))
    return res

def buildD(vd, dims, N):
    res = []
    for i in range(len(dims)):
        res.append(1.0 / 4 / N[i] * vd[i])
    return res

# Selection 1
def calcS(dims, ljk):
    res = []
    for i in range(len(dims)):
            res.append(ls1.calcS(dims[i], ljk[i]))
    return res

def buildS(vs, dims, s, h):
    res = []
    for i in range(len(dims)):
        res.append(s[i] * h[i] * vs[i])
    return res

# Selection 2 (with h != 0.5)
def calcS2(dims, ljk):
    res = []
    for i in range(len(dims)):
        res.append(ls1.calcS2(dims[i], ljk[i]))
    return res

def buildS2(vs, dims, s, h):
    res = []
    for i in range(len(dims)):
        res.append(s[i] * (1-2.0*h[i]) * vs[i])
    return res

#----------------------------------
# updates for the time integration-
#----------------------------------

# 2D
def ud1_2pop_1(sfs, Q, dims):
    for i in range(int(dims[1])):
        sfs[:,i] = Q[0].dot(sfs[:,i])
    return sfs

def ud1_2pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        sfs[i,:] = Q[1].dot(sfs[i,:])
    return sfs

def ud2_2pop_1(sfs, slv, dims):
    for i in range(int(dims[1])):
        sfs[:,i] = slv[0](sfs[:,i])
    return sfs

def ud2_2pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        sfs[i,:] = slv[1](sfs[i,:])
    return sfs

# for 3D, 4D and 5D cases, each couple of directions are coded separately to simplify the permutations...
#------------------------------
# 3D
# step 1

def ud1_3pop_1(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[1])):
            sfs[:, j, i] = Q[0].dot(sfs[:, j, i])
    return sfs

def ud1_3pop_2(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[0])):
            sfs[j, :, i] = Q[1].dot(sfs[j, :, i])
    return sfs

def ud1_3pop_3(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[0])):
            sfs[j, i, :] = Q[2].dot(sfs[j, i, :])
    return sfs

# step 2
def ud2_3pop_1(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[1])):
            sfs[:, j, i] = slv[0](sfs[:, j, i])
    return sfs

def ud2_3pop_2(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[0])):
            sfs[j, :, i] = slv[1](sfs[j, :, i])
    return sfs

def ud2_3pop_3(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[0])):
            sfs[j, i, :] = slv[2](sfs[j, i, :])
    return sfs

#------------------------------
# 4D
# step 1
def ud1_4pop_1(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :] = Q[3].dot(sfs[i, j, k, :])
    return sfs

def ud1_4pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k] = Q[2].dot(sfs[i, j, :, k])
    return sfs

def ud1_4pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k] = Q[1].dot(sfs[i, :, j, k])
    return sfs

def ud1_4pop_4(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k] = Q[0].dot(sfs[:, i, j, k])
    return sfs

# step 2
def ud2_4pop_1(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :] = slv[3](sfs[i, j, k, :])
    return sfs

def ud2_4pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k] = slv[2](sfs[i, j, :, k])
    return sfs

def ud2_4pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k] = slv[1](sfs[i, :, j, k])
    return sfs

def ud2_4pop_4(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k] = slv[0](sfs[:, i, j, k])
    return sfs


#------------------------------
# 5D
# step 1
def ud1_5pop_1(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                for l in range(int(dims[3])):
                    sfs[i, j, k, l, :] = Q[4].dot(sfs[i, j, k, l, :])
    return sfs

def ud1_5pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                for l in range(int(dims[4])):
                    sfs[i, j, k, :, l] = Q[3].dot(sfs[i, j, k, :, l])
    return sfs

def ud1_5pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                for l in range(int(dims[4])):
                    sfs[i, j, :, k, l] = Q[2].dot(sfs[i, j, :, k, l])
    return sfs

def ud1_5pop_4(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                for l in range(int(dims[4])):
                    sfs[i, :, j, k, l] = Q[1].dot(sfs[i, :, j, k, l])
    return sfs

def ud1_5pop_5(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                for l in range(int(dims[4])):
                    sfs[:, i, j, k, l] = Q[0].dot(sfs[:, i, j, k, l])
    return sfs

# step 2
def ud2_5pop_1(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                for l in range(int(dims[3])):
                    sfs[i, j, k, l, :] = slv[4](sfs[i, j, k, l, :])
    return sfs

def ud2_5pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                for l in range(int(dims[4])):
                    sfs[i, j, k, :, l] = slv[3](sfs[i, j, k, :, l])
    return sfs

def ud2_5pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                for l in range(int(dims[4])):
                    sfs[i, j, :, k, l] = slv[2](sfs[i, j, :, k, l])
    return sfs

def ud2_5pop_4(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                for l in range(int(dims[4])):
                    sfs[i, :, j, k, l] = slv[1](sfs[i, :, j, k, l])
    return sfs

def ud2_5pop_5(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                for l in range(int(dims[4])):
                    sfs[:, i, j, k, l] = slv[0](sfs[:, i, j, k, l])
    return sfs

# neutral case - step 2
'''
# 2D
def udn2_2pop_1(sfs, Q1, dims):
    for i in range(int(dims[1])):
        sfs[:,i] = sp.linalg.solve_banded((1, 1), Q1[0], sfs[:,i])
    return sfs

def udn2_2pop_2(sfs, Q1, dims):
    for i in range(int(dims[0])):
        sfs[i,:] = sp.linalg.solve_banded((1, 1), Q1[1], sfs[i,:])
    return sfs

# 3D
def udn2_3pop_1(sfs, Q1, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[1])):
            sfs[:, j, i] = sp.linalg.solve_banded((1, 1), Q1[0], sfs[:, j, i])
    return sfs

def udn2_3pop_2(sfs, Q1, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[0])):
            sfs[j, :, i] = sp.linalg.solve_banded((1, 1), Q1[1], sfs[j, :, i])
    return sfs

def udn2_3pop_3(sfs, Q1, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[0])):
            sfs[j, i, :] = sp.linalg.solve_banded((1, 1), Q1[2], sfs[j, i, :])
    return sfs

# 4D
def udn2_4pop_1(sfs, Q1, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :] = sp.linalg.solve_banded((1, 1), Q1[3], sfs[i, j, k, :])
    return sfs

def udn2_4pop_2(sfs, Q1, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k] = sp.linalg.solve_banded((1, 1), Q1[2], sfs[i, j, :, k])
    return sfs

def udn2_4pop_3(sfs, Q1, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k] = sp.linalg.solve_banded((1, 1), Q1[1], sfs[i, :, j, k])
    return sfs

def udn2_4pop_4(sfs, Q1, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k] = sp.linalg.solve_banded((1, 1), Q1[0], sfs[:, i, j, k])
    return sfs
'''
# sfs update 
def update_step1(sfs, Q, dims):
    assert(len(sfs.shape) == len(dims))
    assert(len(Q) == len(dims))
    for i in range(len(dims)):
        sfs = eval('ud1_'+str(len(dims))+'pop_'+str(i+1)+'(sfs, Q, dims)')
    return sfs

def update_step2(sfs, slv, dims):
    assert(len(sfs.shape) == len(dims))
    assert(len(slv) == len(dims))
    for i in range(len(dims)):
        sfs = eval('ud2_'+str(len(dims))+'pop_'+str(i+1)+'(sfs, slv, dims)')
    return sfs

'''
# neutral case
def update_neutral_step2(sfs, Q1, dims):
    assert(len(sfs.shape) == len(dims))
    assert(len(Q1) == len(dims))
    for i in range(len(dims)):
        sfs = eval('udn2_'+str(len(dims))+'pop_'+str(i+1)+'(sfs, Q1, dims)')
    return sfs
'''
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
    vd = calcD(dims)
    
    # selection part 1
    vs = calcS(dims, ljk)
    S1 = buildS(vs, dims, s, h)
    
    # selection part 2
    vs2 = calcS2(dims, ljk2)
    S2 = buildS2(vs2, dims, s, h)
    
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
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc') - dt/2.0*(D[i]+S1[i]+S2[i])) for i in range(len(n))]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc') + dt/2.0*(D[i]+S1[i]+S2[i]) for i in range(len(n))]
        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs + dt*B)
        elif len(n) > 1:
            sfs = update_step1(sfs, Q, dims)
            sfs = update_step2(sfs + dt*B, slv, dims)
        Nold = N
        t += dt
        # we update the value of N if a function was provided as argument
        if callable(Npop) : N = np.array(Npop(t / 2.0))
    return Spectrum_mod.Spectrum(sfs)


def integrate_nomig_nosel(sfs0, Npop, n, tf, dt_fac = 0.05, theta = 1.0):
    assert(len(n) == 1)
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
    vd = ls1.calcD_dense(dims[0])
    vdb = to_good_shape_banded(vd, 1, 1)

    # mutations
    B = calcB(dims, u)

    id_bd = id_banded(dims[0], 1, 1)

    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        dt_old = dt
        if t+dt > Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold != N).any() or dt != dt_old:
            Dbd = 1.0 / 4 / N[0] * vdb
            D = 1.0 / 4 / N[0] * vd
            # system inversion for backward scheme
            Q1 = id_bd - 0.5*dt*Dbd
            Q2 = np.eye(dims[0]) + 0.5*dt*D
        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = sp.linalg.solve_banded((1, 1), Q1, np.dot(Q2, sfs) + dt*B)
        Nold = N
        t += dt
        # we update the value of N if a function was provided as argument
        if callable(Npop) : N = np.array(Npop(t / 2.0))
    return Spectrum_mod.Spectrum(sfs)
