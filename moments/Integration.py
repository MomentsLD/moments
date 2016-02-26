import numpy as np
import scipy as sp
import time
from scipy.sparse import linalg

import Spectrum_mod
import LinearSystem as ls
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
    if (len(dims)==1): res = [ls.calcD([dims[0]])]
    else:
        res = []
        for i in range(len(dims)):
            for j in range(i+1, len(dims)):
                res.append(ls.calcD([dims[i],dims[j]]))
    return res

def buildD(vd, dims, N):
    if (len(dims)==1): return [1.0/(4*N[0])*vd[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(1.0/(4*N[i])*vd[ctr][0]+1.0/(4*N[j])*vd[ctr][1])
            ctr += 1
    return res

# Selection 1
def calcS(dims,s,h):
    if (len(dims)==1): return [ls.calcS_jk3([dims[0]], [s[0]], [h[0]])]
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(ls.calcS_jk3([dims[i],dims[j]], [s[i], s[j]], [h[i],h[j]]))
    return res

# Selection 2
def calcS2(dims,s,h):
    if (len(dims)==1): return [ls.calcS2_jk3([dims[0]], [s[0]], [h[0]])]
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            res.append(ls.calcS2_jk3([dims[i],dims[j]], [s[i], s[j]], [h[i],h[j]]))
    return res

# Migrations
def calcM(dims,m):
    if (len(dims)==1): return [sp.sparse.coo_matrix(([], ([], [])), shape = (dims[0], dims[0]), dtype = 'float').tocsc()]
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            mbis = np.array([[0,m[i,j]],[m[j,i],0]])
            res.append(ls.calcM_jk3([dims[i],dims[j]],mbis))
    return res

#----------------------------------
# updates for the time integration-
#----------------------------------
# 1D
def ud1_1pop_1(sfs, Q, dims):
    sfs = Q[0].dot(sfs)
    return sfs
    
def ud2_1pop_1(sfs, slv, dims):
    sfs = slv[0](sfs+dt*B)
    return sfs

# 2D
def ud1_2pop_1(sfs, Q, dims):
    sfs = Q[0].dot(sfs.reshape(dims[0]*dims[1])).reshape(dims)
    return sfs

def ud2_2pop_1(sfs, slv, dims):
    sfs = (slv[0](sfs.reshape(dims[0]*dims[1]))).reshape(dims)
    return sfs

# for 3D, 4D and 5D cases, each couple of directions are coded separately to simplify the permutations...
#------------------------------
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


#------------------------------
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


#------------------------------
# 5D
# step 1
def ud1_5pop_1(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i,j,k,:,:] = Q[9].dot(sfs[i,j,k,:,:].reshape(dims[3]*dims[4])).reshape(dims[3],dims[4])
    return sfs

def ud1_5pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i,j,:,k,:] = Q[8].dot(sfs[i,j,:,k,:].reshape(dims[2]*dims[4])).reshape(dims[2],dims[4])
    return sfs

def ud1_5pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[4])):
                sfs[i,j,:,:,k] = Q[7].dot(sfs[i,j,:,:,k].reshape(dims[2]*dims[3])).reshape(dims[2],dims[3])
    return sfs

def ud1_5pop_4(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i,:,j,k,:] = Q[6].dot(sfs[i,:,j,k,:].reshape(dims[1]*dims[4])).reshape(dims[1],dims[4])
    return sfs

def ud1_5pop_5(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[i,:,j,:,k] = Q[5].dot(sfs[i,:,j,:,k].reshape(dims[1]*dims[3])).reshape(dims[1],dims[3])
    return sfs

def ud1_5pop_6(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[i,:,:,j,k] = Q[4].dot(sfs[i,:,:,j,k].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

def ud1_5pop_7(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:,i,j,k,:] = Q[3].dot(sfs[:,i,j,k,:].reshape(dims[0]*dims[4])).reshape(dims[0],dims[4])
    return sfs

def ud1_5pop_8(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[:,i,j,:,k] = Q[2].dot(sfs[:,i,j,:,k].reshape(dims[0]*dims[3])).reshape(dims[0],dims[3])
    return sfs

def ud1_5pop_9(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:,i,:,j,k] = Q[1].dot(sfs[:,i,:,j,k].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def ud1_5pop_10(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:,:,i,j,k] = Q[0].dot(sfs[:,:,i,j,k].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

# step 2
def ud2_5pop_1(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i,j,k,:,:] = slv[9](sfs[i,j,k,:,:].reshape(dims[3]*dims[4])).reshape(dims[3],dims[4])
    return sfs

def ud2_5pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i,j,:,k,:] = slv[8](sfs[i,j,:,k,:].reshape(dims[2]*dims[4])).reshape(dims[2],dims[4])
    return sfs

def ud2_5pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[4])):
                sfs[i,j,:,:,k] = slv[7](sfs[i,j,:,:,k].reshape(dims[2]*dims[3])).reshape(dims[2],dims[3])
    return sfs

def ud2_5pop_4(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i,:,j,k,:] = slv[6](sfs[i,:,j,k,:].reshape(dims[1]*dims[4])).reshape(dims[1],dims[4])
    return sfs

def ud2_5pop_5(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[i,:,j,:,k] = slv[5](sfs[i,:,j,:,k].reshape(dims[1]*dims[3])).reshape(dims[1],dims[3])
    return sfs

def ud2_5pop_6(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[i,:,:,j,k] = slv[4](sfs[i,:,:,j,k].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
    return sfs

def ud2_5pop_7(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:,i,j,k,:] = slv[3](sfs[:,i,j,k,:].reshape(dims[0]*dims[4])).reshape(dims[0],dims[4])
    return sfs

def ud2_5pop_8(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[:,i,j,:,k] = slv[2](sfs[:,i,j,:,k].reshape(dims[0]*dims[3])).reshape(dims[0],dims[3])
    return sfs

def ud2_5pop_9(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:,i,:,j,k] = slv[1](sfs[:,i,:,j,k].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
    return sfs

def ud2_5pop_10(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:,:,i,j,k] = slv[0](sfs[:,:,i,j,k].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
    return sfs

# update nD with permutations
def update_step1(sfs, Q, dims, order):
    assert(len(sfs.shape)==len(dims))
    assert(len(Q)==len(dims)*(len(dims)-1)/2)
    for i in order:
        sfs = eval('ud1_'+str(len(dims))+'pop_'+str(i+1)+'(sfs, Q, dims)')
    return sfs

def update_step2(sfs, slv, dims, order):
    assert(len(sfs.shape)==len(dims))
    assert(len(slv)==len(dims)*(len(dims)-1)/2)
    for i in order:
        sfs = eval('ud2_'+str(len(dims))+'pop_'+str(i+1)+'(sfs, slv, dims)')
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
# where t is the relative time in generations such as t = 0 initially
# Npop is a lambda function of the time t returning the vector N = (N1,...,Np) or directly the vector if N does not evolve in time
'''def integrate2(sfs0, Npop, n, tf, dt_fac=0.01, gamma=None, h=None, m=None, theta=1.0):
    
    # neutral case if the parameters are not provided
    if gamma is None : gamma = np.zeros(len(n))
    if h is None : h = 0.5*np.ones(len(n))
    if m is None : m = np.zeros([len(n), len(n)])
    
    sfs0 = np.array(sfs0)
    # parameters of the equation
    if callable(Npop): N = np.array(Npop(0))
    else : N = np.array(Npop)
    
    N0=N[0]
    Nold = N+np.ones(len(N))
    mm = np.array(m)/(2.0*N0)
    s = np.array(gamma)/N0
    h = np.array(h)
    Tmax = tf*2.0*N0
    dt = dt_fac*Tmax
    print(dt, Tmax)
    u = theta/(4.0*N0)
    # dimensions of the sfs
    dims = n+np.ones(len(n))
    d = int(np.prod(dims))
    # number of "directions" for the splitting
    nbp = int(len(n)*(len(n)-1)/2)
    if len(n)==1: nbp = 1
    # we compute the matrices we will need
    vd = calcD(dims)
    S1 = calcS(dims,s,h)
    S2 = calcS2(dims,s,h)
    Mi = calcM(dims,mm)
    B = calcB(dims, u)

    # indexes for the permutation trick
    order = list(range(nbp))
    # time step splitting
    split_dt = 1.0
    if len(n)>2: split_dt = 3.0
    if len(n)==5: split_dt = 5.0
    
    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any():
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt/2/split_dt*(1.0/(max(len(n),2)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(nbp)]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')+dt/2/split_dt*(1.0/(max(len(n),2)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in range(nbp)]
        
        # drift, selection and migration (depends on the dimension)
        if len(n)==1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs+dt*B)
        if len(n)>1:
            for i in range(int(split_dt)):
                sfs = update_step1(sfs, Q, dims, order)
                sfs = update_step2(sfs+dt/split_dt*B, slv, dims, order)
                order = permute(order)
        Nold = N
        t += dt
        # we update the value of N if a function was provided as argument
        if callable(Npop) : N = np.array(Npop(t/(2.0*N0)))
    return Spectrum_mod.Spectrum(sfs)'''

def integrate(sfs0, Npop, n, tf, dt_fac=0.05, gamma=None, h=None, m=None, theta=1.0):
    # neutral case if the parameters are not provided
    if gamma is None : gamma = np.zeros(len(n))
    if h is None : h = 0.5*np.ones(len(n))
    if m is None : m = np.zeros([len(n), len(n)])
    
    sfs0 = np.array(sfs0)
    # parameters of the equation
    if callable(Npop): N = np.array(Npop(0))
    else : N = np.array(Npop)
    
    Nold = N+np.ones(len(N))
    mm = np.array(m)/2.0
    s = np.array(gamma)
    h = np.array(h)
    Tmax = tf*2.0
    dt = Tmax*dt_fac
    u = theta/4.0
    # dimensions of the sfs
    dims = n+np.ones(len(n))
    d = int(np.prod(dims))
    # number of "directions" for the splitting
    nbp = int(len(n)*(len(n)-1)/2)
    if len(n)==1: nbp = 1
    # we compute the matrices we will need
    vd = calcD(dims)
    S1 = calcS(dims,s,h)
    S2 = calcS2(dims,s,h)
    Mi = calcM(dims,mm)
    B = calcB(dims, u)
    
    # indexes for the permutation trick
    order = list(range(nbp))
    # time step splitting
    split_dt = 1.0
    if len(n)>2: split_dt = 3.0
    if len(n)==5: split_dt = 5.0
    
    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        dt_old = dt
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any() or dt != dt_old:
            D = buildD(vd, dims, N)
            
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')-dt/2.0/split_dt*(1.0/(max(len(n),2)-1)*(D[i]+S1[i]+S2[i])+Mi[i])) for i in range(nbp)]
            Q = [sp.sparse.identity(S1[i].shape[0], dtype = 'float', format = 'csc')+dt/2.0/split_dt*(1.0/(max(len(n),2)-1)*(D[i]+S1[i]+S2[i])+Mi[i]) for i in range(nbp)]
        
        # drift, selection and migration (depends on the dimension)
        if len(n)==1:
            sfs = Q[0].dot(sfs)
            sfs = slv[0](sfs+dt*B)
        if len(n)>1:
            for i in range(int(split_dt)):
                sfs = update_step1(sfs, Q, dims, order)
                sfs = update_step2(sfs+dt/split_dt*B, slv, dims, order)
                order = permute(order)
        Nold = N
        t += dt
        # we update the value of N if a function was provided as argument
        if callable(Npop) : N = np.array(Npop(t/2.0))
    return Spectrum_mod.Spectrum(sfs)



