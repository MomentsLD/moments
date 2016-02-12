import numpy as np
import scipy as sp
from scipy.sparse import linalg
import math

import time

import jackknife as jk
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

# Mutations
def mutate(sfs, u, dims, dt):
    for i in range(len(dims)):
        ind = np.zeros(len(dims), dtype='int')
        ind[i] = int(1)
        sfs[tuple(ind)] += dt*u*(dims[i]-1)
    return sfs

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
            #print(i,j)
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

# updates for the time integration
# 2D
def update_2popCN_step1(sfs, Q, dims):
    assert(len(sfs.shape)==2)
    assert(len(Q)==1)
    sfs = Q[0].dot(sfs.reshape(dims[0]*dims[1])).reshape(dims)
    return sfs

def update_2pop(sfs, slv, dims):
    assert(len(sfs.shape)==2)
    assert(len(slv)==1)
    sfs = (slv[0](sfs.reshape(dims[0]*dims[1]))).reshape(dims)
    return sfs

# 3D

# geerating the indexing structure for the permutation trick
def gen_index(d):
    index = [[i,j] for i in range(d) for j in range(i+1,d)]
    order = list(range(d*(d-1)/2))
    ind_loop = []
    for i in range(d*(d-1)/2):
        l = list(range(d))
        l.remove(index[i][0])
        l.remove(index[i][1])
        ind_loop.append(l)
    return index[::-1],order[::-1], ind_loop[::-1]

def update_3popCN_step1_bis(sfs, Q, dims, index, order, ind_loop):
    assert(len(sfs.shape)==3)
    assert(len(Q)==3)
    for i in range(int(ind_loop[0])):
        sfs[i,:,:] = Q[order[0]].dot(sfs[i,:,:].reshape(dims[int(index[0][0])]*dims[int(index[0][1])])).reshape(dims[int(index[0][0])],dims[int(index[0][1])])
    for i in range(int(ind_loop[1])):
        sfs[:,i,:] = Q[order[1]].dot(sfs[:,i,:].reshape(dims[int(index[1][0])]*dims[int(index[1][1])])).reshape(dims[int(index[1][0])],dims[int(index[1][1])])
    for i in range(int(ind_loop[2])):
        sfs[:,:,i] = Q[order[2]].dot(sfs[:,:,i].reshape(dims[int(index[2][0])]*dims[int(index[2][1])])).reshape(dims[int(index[2][0])],dims[int(index[2][1])])
    return sfs

def update_3pop_bis(sfs, slv, dims, index, order, ind_loop):
    assert(len(sfs.shape)==3)
    assert(len(slv)==3)
    for i in range(int(ind_loop[0])):
        sfs[i,:,:] = slv[order[0]](sfs[i,:,:].reshape(dims[int(index[0][0])]*dims[int(index[0][1])])).reshape(dims[int(index[0][0])],dims[int(index[0][1])])
    for i in range(int(ind_loop[1])):
        sfs[:,i,:] = slv[order[1]](sfs[:,i,:].reshape(dims[int(index[1][0])]*dims[int(index[1][1])])).reshape(dims[int(index[1][0])],dims[int(index[1][1])])
    for i in range(int(ind_loop[2])):
        sfs[:,:,i] = slv[order[2]](sfs[:,:,i].reshape(dims[int(index[2][0])]*dims[int(index[2][1])])).reshape(dims[int(index[2][0])],dims[int(index[2][1])])
    return sfs

def update_3popCN_step1(sfs, Q, dims):
    assert(len(sfs.shape)==3)
    assert(len(Q)==3)
    for i in range(int(dims[0])):
        sfs[i,:,:] = Q[2].dot(sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        sfs[:,i,:] = Q[1].dot(sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        sfs[:,:,i] = Q[0].dot(sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

def update_3pop(sfs, slv, dims):
    assert(len(sfs.shape)==3)
    assert(len(slv)==3)
    for i in range(int(dims[0])):
        sfs[i,:,:] = slv[2](sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        sfs[:,i,:] = slv[1](sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        sfs[:,:,i] = slv[0](sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

def update_3popCN_step12(sfs, Q, dims):
    assert(len(sfs.shape)==3)
    assert(len(Q)==3)
    for i in range(int(dims[1])):
        sfs[:,i,:] = Q[1].dot(sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        sfs[:,:,i] = Q[0].dot(sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    for i in range(int(dims[0])):
        sfs[i,:,:] = Q[2].dot(sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

def update_3pop2(sfs, slv, dims):
    assert(len(sfs.shape)==3)
    assert(len(slv)==3)
    for i in range(int(dims[1])):
        sfs[:,i,:] = slv[1](sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        sfs[:,:,i] = slv[0](sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    for i in range(int(dims[0])):
        sfs[i,:,:] = slv[2](sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

def update_3popCN_step13(sfs, Q, dims):
    assert(len(sfs.shape)==3)
    assert(len(Q)==3)
    for i in range(int(dims[2])):
        sfs[:,:,i] = Q[0].dot(sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    for i in range(int(dims[0])):
        sfs[i,:,:] = Q[2].dot(sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        sfs[:,i,:] = Q[1].dot(sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def update_3pop3(sfs, slv, dims):
    assert(len(sfs.shape)==3)
    assert(len(slv)==3)
    for i in range(int(dims[2])):
        sfs[:,:,i] = slv[0](sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    for i in range(int(dims[0])):
        sfs[i,:,:] = slv[2](sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        sfs[:,i,:] = slv[1](sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def update_3popCN_step1_reverse(sfs, Q, dims):
    assert(len(sfs.shape)==3)
    assert(len(Q)==3)
    for i in range(int(dims[0])):
        sfs[i,:,:] = Q[2].dot(sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        sfs[:,i,:] = Q[1].dot(sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        sfs[:,:,i] = Q[0].dot(sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

def update_3pop_reverse(sfs, slv, dims):
    assert(len(sfs.shape)==3)
    assert(len(slv)==3)
    for i in range(int(dims[0])):
        sfs[i,:,:] = slv[2](sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        sfs[:,i,:] = slv[1](sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        sfs[:,:,i] = slv[0](sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs


# 4D
def update_4popCN_step1(sfs, Q, dims):
    assert(len(sfs.shape)==4)
    assert(len(Q)==6)
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i,j,:,:] = Q[5].dot(sfs[i,j,:,:].reshape(dims[2]*dims[3])).reshape(dims[2],dims[3])
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i,:,j,:] = Q[4].dot(sfs[i,:,j,:].reshape(dims[1]*dims[3])).reshape(dims[1],dims[3])
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i,:,:,j] = Q[3].dot(sfs[i,:,:,j].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:,i,j,:] = Q[2].dot(sfs[:,i,j,:].reshape(dims[0]*dims[3])).reshape(dims[0],dims[3])
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:,i,:,j] = Q[1].dot(sfs[:,i,:,j].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:,:,i,j] = Q[0].dot(sfs[:,:,i,j].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs


def update_4pop(sfs, slv, dims):
    assert(len(sfs.shape)==4)
    assert(len(slv)==6)
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i,j,:,:] = slv[5](sfs[i,j,:,:].reshape(dims[2]*dims[3])).reshape(dims[2],dims[3])
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i,:,j,:] = slv[4](sfs[i,:,j,:].reshape(dims[1]*dims[3])).reshape(dims[1],dims[3])
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i,:,:,j] = slv[3](sfs[i,:,:,j].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:,i,j,:] = slv[2](sfs[:,i,j,:].reshape(dims[0]*dims[3])).reshape(dims[0],dims[3])
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:,i,:,j] = slv[1](sfs[:,i,:,j].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:,:,i,j] = slv[0](sfs[:,:,i,j].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
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
# fctN is the name of a "lambda" fuction giving N = fctN(t)
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

    # we compute the matrices we will need
    vd = calcD(dims)
    S1 = calcS(dims,s,h)
    S2 = calcS2(dims,s,h)
    Mi = calcM_jk3(dims,mm)
    
    interval = time.time() - start_time
    print('Time init:', interval)
    start_time = time.time()

    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any():
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(len(n)*(len(n)-1)/2)]

        # Backward Euler scheme with splitted operators
        
        # mutations
        sfs = mutate(sfs, u, dims, dt)
        # drift, selection and migration (depends on the dimension)
        if len(dims)==1 : sfs = slv[0](sfs)
        if len(dims)==2 : sfs = update_2pop(sfs, slv, dims)
        if len(dims)==3 : sfs = update_3pop(sfs, slv, dims)
        if len(dims)==4 : sfs = update_4pop(sfs, slv, dims)
    
        Nold = N
        t += dt
        N = np.array(fctN(t/(2.0*N0)))
    
    interval = time.time() - start_time
    print('Time loop:', interval)

    return sfs

def integrateCN(sfs0, fctN, n, tf, dt, gamma, h, m, theta=1.0):
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
    
    # we compute the matrices we will need
    vd = calcD(dims)
    S1 = calcS(dims,s,h)
    S2 = calcS2(dims,s,h)
    Mi = calcM_jk3(dims,mm)
    B = calcB(dims, u)
    interval = time.time() - start_time
    print('Time init:', interval)
    start_time = time.time()
    
    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any():
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt/2/3*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(len(n)*(len(n)-1)/2)]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')+dt/2/3*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in range(len(n)*(len(n)-1)/2)]
            #slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt/2*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(len(n)*(len(n)-1)/2)]
            #Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')+dt/2*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in range(len(n)*(len(n)-1)/2)]
        # Backward Euler scheme with splitted operators
        
        # mutations
        # drift, selection and migration (depends on the dimension)
        if len(dims)==1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs+dt*B)
        if len(dims)==2:
            sfs = update_2popCN_step1(sfs, Q, dims)
            sfs = update_2pop(sfs+dt*B, slv, dims)
        if len(dims)==3:
            sfs = update_3popCN_step1(sfs, Q, dims)
            sfs = update_3pop(sfs+dt/3*B, slv, dims)
            sfs = update_3popCN_step12(sfs, Q, dims)
            sfs = update_3pop2(sfs+dt/3*B, slv, dims)
            sfs = update_3popCN_step13(sfs, Q, dims)
            sfs = update_3pop3(sfs+dt/3*B, slv, dims)
        #sfs = update_3popCN_all(sfs, B, Q, slv, dims, dt)
        if len(dims)==4:
            sfs = update_4popCN_step1(sfs, Q, dims)
            sfs = update_4pop(sfs+dt*B, slv, dims)
        
        Nold = N
        t += dt
        N = np.array(fctN(t/(2.0*N0)))
    
    interval = time.time() - start_time
    print('Time loop:', interval)

    return sfs

def integrateCN2(sfs0, fctN, n, tf, dt, gamma, h, m, theta=1.0):
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
    index, order, ind_lp = gen_index(len(n))
    
    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any():
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt/2/3*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in order]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')+dt/2/3*(1.0/(len(n)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in order]
        
        # mutations
        # drift, selection and migration (depends on the dimension)
        if len(dims)==1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs+dt*B)
        if len(dims)==2:
            sfs = update_2popCN_step1(sfs, Q, dims)
            sfs = update_2pop(sfs+dt*B, slv, dims)
        if len(dims)==3:
            for i in range(3):
                sfs = update_3popCN_step1_bis(sfs, Q, dims, index, order, ind_lp)
                sfs = update_3pop_bis(sfs+dt/3*B, slv, dims, index, order, ind_lp)
                Q = permute(Q)
                slv = permute(slv)
                index = permute(index)
                order = permute(order)
                ind_lp = permute(ind_lp)
        if len(dims)==4:
            sfs = update_4popCN_step1(sfs, Q, dims)
            sfs = update_4pop(sfs+dt*B, slv, dims)
        
        Nold = N
        t += dt
        N = np.array(fctN(t/(2.0*N0)))
    
    interval = time.time() - start_time
    print('Time loop:', interval)

    return sfs
