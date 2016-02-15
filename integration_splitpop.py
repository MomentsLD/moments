import numpy as np
import scipy as sp
from scipy.sparse import linalg
import math

import time

import integration_multiD_sparse as its
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
        ind = np.zeros(len(dims), dtype='int')
        ind[k] = int(1)
        tp = tuple(ind)
        B[tp] = dims[k]-1
    return u*B

# Drift
def calcD(dims):
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(its.calcD([dims[i],dims[j]]))
    return res

def buildD(vd, dims, N):
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(1.0/(4*N[i])*vd[ctr][0]+1.0/(4*N[j])*vd[ctr][1])
            ctr += 1
    return res

# Selection 1
def calcS(dims,s,h):
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(its.calcS_jk3([dims[i],dims[j]], [s[i], s[j]], [h[i],h[j]]))
    return res

# Selection 2
def calcS2(dims,s,h):
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(its.calcS2_jk3([dims[i],dims[j]], [s[i], s[j]], [h[i],h[j]]))
    return res

# Migrations
def calcM_jk3(dims,m):
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            mbis = np.array([[0,m[i,j]],[m[j,i],0]])
            res.append(its.calcM_jk3([dims[i],dims[j]],mbis))
    return res

#----------------------------------
# updates for the time integration-
#----------------------------------
# 2D
def update1_2pop(sfs, Q, dims):
    assert(len(sfs.shape)==2)
    assert(len(Q)==1)
    sfs = Q[0].dot(sfs.reshape(dims[0]*dims[1])).reshape(dims)
    return sfs

def update2_2pop(sfs, slv, dims):
    assert(len(sfs.shape)==2)
    assert(len(slv)==1)
    sfs = (slv[0](sfs.reshape(dims[0]*dims[1]))).reshape(dims)
    return sfs

# for 3D and 4D cases, each couple of directions are coded separately to simplify the permutations...

# 3D
# step 1
def ud1_3pop_1(sfs, Q, dims):
    for i in range(int(dims[2])):
        sfs[:,:,i] = Q[0].dot(sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

def ud1_3pop_2(sfs, Q, dims):
    for i in range(int(dims[1])):
        sfs[:,i,:] = Q[1].dot(sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def ud1_3pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        sfs[i,:,:] = Q[2].dot(sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

# step 2
def ud2_3pop_1(sfs, slv, dims):
    for i in range(int(dims[2])):
        sfs[:,:,i] = slv[0](sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

def ud2_3pop_2(sfs, slv, dims):
    for i in range(int(dims[1])):
        sfs[:,i,:] = slv[1](sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def ud2_3pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        sfs[i,:,:] = slv[2](sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

# update 3D with permutations
def update1_3pop(sfs, Q, dims, order = range(3)):
    assert(len(sfs.shape)==3)
    assert(len(Q)==3)
    for i in order:
        sfs = eval('ud1_3pop_'+str(i+1)+'(sfs, Q, dims)')
    return sfs

def update2_3pop(sfs, slv, dims, order = range(3)):
    assert(len(sfs.shape)==3)
    assert(len(slv)==3)
    for i in order:
        sfs = eval('ud2_3pop_'+str(i+1)+'(sfs, slv, dims)')
    return sfs


# 4D
# step 1
def ud1_4pop_1(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i,j,:,:] = Q[5].dot(sfs[i,j,:,:].reshape(dims[2]*dims[3])).reshape(dims[2],dims[3])
    return sfs

def ud1_4pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i,:,j,:] = Q[4].dot(sfs[i,:,j,:].reshape(dims[1]*dims[3])).reshape(dims[1],dims[3])
    return sfs

def ud1_4pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i,:,:,j] = Q[3].dot(sfs[i,:,:,j].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

def ud1_4pop_4(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:,i,j,:] = Q[2].dot(sfs[:,i,j,:].reshape(dims[0]*dims[3])).reshape(dims[0],dims[3])
    return sfs

def ud1_4pop_5(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:,i,:,j] = Q[1].dot(sfs[:,i,:,j].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def ud1_4pop_6(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:,:,i,j] = Q[0].dot(sfs[:,:,i,j].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

# step 2
def ud2_4pop_1(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i,j,:,:] = slv[5](sfs[i,j,:,:].reshape(dims[2]*dims[3])).reshape(dims[2],dims[3])
    return sfs

def ud2_4pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i,:,j,:] = slv[4](sfs[i,:,j,:].reshape(dims[1]*dims[3])).reshape(dims[1],dims[3])
    return sfs

def ud2_4pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i,:,:,j] = slv[3](sfs[i,:,:,j].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

def ud2_4pop_4(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:,i,j,:] = slv[2](sfs[:,i,j,:].reshape(dims[0]*dims[3])).reshape(dims[0],dims[3])
    return sfs

def ud2_4pop_5(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:,i,:,j] = slv[1](sfs[:,i,:,j].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def ud2_4pop_6(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:,:,i,j] = slv[0](sfs[:,:,i,j].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

# update 4D with permutations
def update1_4pop(sfs, Q, dims, order = range(6)):
    assert(len(sfs.shape)==4)
    assert(len(Q)==6)
    for i in order:
        sfs = eval('ud1_4pop_'+str(i+1)+'(sfs, Q, dims)')
    return sfs


def update2_4pop(sfs, slv, dims, order = range(6)):
    assert(len(sfs.shape)==4)
    assert(len(slv)==6)
    for i in order:
        sfs = eval('ud2_4pop_'+str(i+1)+'(sfs, slv, dims)')
    return sfs

def permute(tab):
    res = tab[1:]
    res.append(tab[0])
    return res
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
# fctN is the name of a "lambda" function giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
# fctN is a lambda function of the time t returning the vector N = (N1,...,Np)
def integrate(sfs0, fctN, n, tf, dt, gamma, h, m, theta=1.0):
    # parameters of the equation
    start_time = time.time()
    
    N = np.array(fctN(0))
    N0=N[0]
    Nold = N+np.ones(len(N))
    mm = np.array(m)/(2.0*N0)
    s = np.array(gamma)/N0
    h = np.array(h)
    Tmax = tf*2.0*N0
    dt = dt*2.0*N0
    u = theta/(4.0*N0)
    # dimensions of the sfs
    dims = n+np.ones(len(n))
    d = int(np.prod(dims))
    # number of "directions" for the splitting
    nbp = len(n)*(len(n)-1)/2
    
    # we compute the matrices we will need
    vd = calcD(dims)
    S1 = calcS(dims,s,h)
    S2 = calcS2(dims,s,h)
    Mi = calcM_jk3(dims,mm)
    B = calcB(dims, u)
    interval = time.time() - start_time
    print('Time init:', interval)
    start_time = time.time()
    
    # indexes for the permutation trick
    order = list(range(nbp))
    
    # time step splitting
    split_dt = 1.0
    if len(n)>2: split_dt = 3.0
    
    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any():
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt/2/split_dt*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(nbp)]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')+dt/2/split_dt*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in range(nbp)]

        # drift, selection and migration (depends on the dimension)
        if len(dims)==1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs+dt*B)
        if len(dims)==2:
            sfs = update1_2pop(sfs, Q, dims)
            sfs = update2_2pop(sfs+dt*B, slv, dims)
        if len(dims)==3:
            for i in range(int(split_dt)):
                sfs = update1_3pop(sfs, Q, dims, order)
                sfs = update2_3pop(sfs+dt/split_dt*B, slv, dims, order)
                order = permute(order)
        if len(dims)==4:
            for i in range(int(split_dt)):
                sfs = update1_4pop(sfs, Q, dims, order)
                sfs = update2_4pop(sfs+dt/split_dt*B, slv, dims, order)
                order = permute(order)

        Nold = N
        t += dt
        N = np.array(fctN(t/(2.0*N0)))

    interval = time.time() - start_time
    print('Time loop:', interval)

    return sfs

