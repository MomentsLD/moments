import numpy as np
import scipy as sp
from scipy.sparse import linalg
import math

import time

import jackknife as jk
import integration_multiD as itd
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
#-------------------------------------------------------------------------------

# Function that returns the 1D index corresponding to ind = [i1,...,ip] when using reshape
# dims = numpy.array([n1,...np])
def index_1D(ind, dims):
    res = 0
    for i in range(len(dims)):
        f = 1
        for j in range(len(dims)-i-1):
            f *= dims[len(dims)-1-j]
        res += f*ind[i]
    return res

# Computes the n-dimensional index from the 1D index (inverse of index_1D)
def index_nD(id, dims):
    res = []
    r = id
    for i in range(len(dims)):
        f = 1
        for j in range(len(dims)-i-1):
            f *= dims[len(dims)-1-j]
        res.append(r//f)
        r = r%f
    return np.array(res)


# Mutations
def calcB(u, dims):
    B = np.zeros(dims)
    for k in range(len(dims)):
        ind = np.zeros(len(dims), dtype='int')
        ind[k] = int(1)
        tp = tuple(ind)
        B[tp] = dims[k]-1
    return u*B
def mutate(sfs, u, dims, dt):
    for i in range(len(dims)):
        ind = np.zeros(len(dims), dtype='int')
        ind[i] = int(1)
        sfs[tuple(ind)] += dt*u*(dims[i]-1)
    return sfs

# We compute the  matrices for drift
# this function returns a list of matrices corresponding to each population
# dims -> array containing the dimensions of the problem dims[j] = nj+1
def calcD(dims):
    # we consider separately the contributions of each dimension
    res = []
    for j in range(len(dims)):
        data = []
        row = []
        col = []
        # loop over the fs elements:
        for i in range(int(dims[j])):
            if (i>1):
                data.append((i-1)*(dims[j]-i))
                row.append(i)
                col.append(i-1)
            if (i<dims[j]-2):
                data.append((i+1)*(dims[j]-i-2))
                col.append(i+1)
                row.append(i)
            if (i>0) and (i<dims[j]-1):
                data.append(-2*i*(dims[j]-i-1))
                row.append(i)
                col.append(i)
        res.append(sp.sparse.coo_matrix((data, (row, col)), shape = (dims[j], dims[j]), dtype = 'float').tocsc())
    return res

# Selection
# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
# with order 3 JK...
def calcS_jk3(dims):
    # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK13(int(dims[i]-1)))
    
    res = []

    for j in range(len(dims)):
        data = []
        row = []
        col = []
        for i in range(int(dims[j])):
            g1 = 1.0/dims[j]*i*(dims[j]-i)
            g2 = -1.0/dims[j]*(i+1)*(dims[j]-1-i)

            ibis = jk.index_bis(i,dims[j]-1)
            iter = jk.index_bis(i+1,dims[j]-1)

            if (i<dims[j]-1):
                data += [g1*ljk[j][i-1,ibis-2], g1*ljk[j][i-1,ibis-1],
                         g1*ljk[j][i-1,ibis], g2*ljk[j][i,iter-2],
                         g2*ljk[j][i,iter-1], g2*ljk[j][i,iter]]
                row += [i]*6
                col += [ibis-1, ibis, ibis+1, iter-1, iter, iter+1]
            
            if i==dims[j]-1: # g2=0
                data += [g1*ljk[j][i-1,ibis-2], g1*ljk[j][i-1,ibis-1], g1*ljk[j][i-1,ibis]]
                row += [i]*3
                col += [ibis-1, ibis, ibis+1]
        res.append(sp.sparse.coo_matrix((data, (row, col)), shape = (dims[j], dims[j]), dtype = 'float').tocsc())
    return res


# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
def calcS2_jk3(dims):
    # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK23(int(dims[i]-1)))

    res= []
    for j in range(len(dims)):
        data = []
        row = []
        col = []
        for i in range(int(dims[j])):
            g1 = 1.0*(i+1)/dims[j]/(dims[j]+1)*i*(dims[j]-i)
            g2 = -1.0*(i+1)/dims[j]/(dims[j]+1)*(i+2)*(dims[j]-1-i)
            iter = jk.index_bis(i+1,dims[j]-1)
            iqua = jk.index_bis(i+2,dims[j]-1)

            if i<dims[j]-1:
                data += [g1*ljk[j][i,iter-2], g1*ljk[j][i,iter-1],
                         g1*ljk[j][i,iter], g2*ljk[j][i+1,iqua-2],
                         g2*ljk[j][i+1,iqua-1], g2*ljk[j][i+1,iqua]]
                row += [i]*6
                col += [iter-1, iter, iter+1,
                        iqua-1, iqua, iqua+1]
            
            if i==dims[j]-1: # g2=0
                data += [g1*ljk[j][i,iter-2], g1*ljk[j][i,iter-1], g1*ljk[j][i,iter]]
                row += [i]*3
                col += [iter-1, iter, iter+1]
        res.append(sp.sparse.coo_matrix((data, (row, col)), shape = (dims[j], dims[j]), dtype = 'float').tocsc())
    return res
# Migrate...
def migrate(sfs, mi, dt):
    dims = sfs.shape
    d = int(np.prod(dims))
    #nd = len(dims)
    fs = sfs.copy()
    fs = fs.reshape(d)
    slv = linalg.factorized(sp.sparse.identity(d, dtype = 'float', format = 'csc')-dt*mi)
    fs = slv(fs)
    fs = fs.reshape(dims)
    return fs

# Migration
# m -> migration rates matrix, m[i,j] = migration rate from pop i to pop j
# with order 3 JK
def calcM_jk3(dims, m):
    # number of degrees of freedom
    d = int(np.prod(dims))
    
    # we don't compute the matrix if not necessary
    if (len(dims)==1) : return  sp.sparse.coo_matrix(([], ([], [])), shape = (dims[0], dims[0]), dtype = 'float').tocsc()
    if (not m.any()) : return  sp.sparse.coo_matrix(([], ([], [])), shape = (d, d), dtype = 'float').tocsc()
   
   # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK13(int(dims[i]-1)))

    data = []
    row = []
    col = []
    for i in range(d):
        # multi-D index of the current variable
        index = index_nD(i, dims)
        
        for j in range(len(dims)):
    
            indj = np.zeros(len(dims), dtype='int')
            indj[j] = int(1)
            
            index_bisj = np.array(index)
            index_bisj[j] = jk.index_bis(index_bisj[j],dims[j]-1)
            index_terj = np.array(index)+indj
            index_terj[j] = jk.index_bis(index_terj[j],dims[j]-1)
            
            coeff1 = (2*index[j]-(dims[j]-1))/(dims[j]-1)
            coeff2 = (dims[j]-index[j])/(dims[j]-1)
            coeff3 = -(index[j]+1)/(dims[j]-1)
            for k in range(len(dims)):
                if k != j:
                    indk = np.zeros(len(dims), dtype='int')
                    indk[k] = int(1)
                    
                    c = (dims[j]-1)*(index[k]+1)/dims[k]
                    
                    data.append(-m[j,k]*index[j])
                    row.append(i)
                    col.append(i)
                    
                    index_bisk = np.array(index)
                    index_bisk[k] = jk.index_bis(index_bisk[k],dims[k]-1)
                    index_terk = np.array(index)+indk
                    index_terk[k] = jk.index_bis(index_terk[k],dims[k]-1)
                    
                    if index[j] < dims[j]-1:
                        data.append(m[j,k]*(index[j]+1))
                        row.append(i)
                        col.append(index_1D(index+indj, dims))
                    
                    if index[k] < dims[k]-1:
                        data += [m[j,k]*coeff1*ljk[k][index[k],index_terk[k]-2]*c, m[j,k]*coeff1*ljk[k][index[k],index_terk[k]-1]*c, m[j,k]*coeff1*ljk[k][index[k],index_terk[k]]*c]
                        row += [i]*3
                        col += [index_1D(index_terk-indk, dims), index_1D(index_terk, dims), index_1D(index_terk+indk, dims)]
                        if index[j] > 0:
                            data +=[m[j,k]*coeff2*ljk[k][index[k],index_terk[k]-2]*c, m[j,k]*coeff2*ljk[k][index[k],index_terk[k]-1]*c, m[j,k]*coeff2*ljk[k][index[k],index_terk[k]]*c]
                            row += [i]*3
                            col += [index_1D(index_terk-indk-indj, dims), index_1D(index_terk-indj, dims), index_1D(index_terk+indk-indj, dims)]
                        if index[j] < dims[j]-1:
                            data += [m[j,k]*coeff3*ljk[k][index[k],index_terk[k]-2]*c, m[j,k]*coeff3*ljk[k][index[k],index_terk[k]-1]*c, m[j,k]*coeff3*ljk[k][index[k],index_terk[k]]*c]
                            row += [i]*3
                            col += [index_1D(index_terk-indk+indj, dims), index_1D(index_terk+indj, dims), index_1D(index_terk+indk+indj, dims)]
                            
                    if index[k] == dims[k]-1:
                        data += [m[j,k]*coeff1*c, m[j,k]*coeff1*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]-2]*c,
                                 m[j,k]*coeff1*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]-1]*c,
                                 m[j,k]*coeff1*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]]*c]
                        row += [i]*4
                        col += [i, index_1D(index_terk-indk, dims), index_1D(index_terk, dims), index_1D(index_terk+indk, dims)]

                        if index[j] > 0:
                            data += [m[j,k]*coeff2*c, m[j,k]*coeff2*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]-2]*c,
                                     m[j,k]*coeff2*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]-1]*c,
                                     m[j,k]*coeff2*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]]*c]
                            row += [i]*4
                            col += [index_1D(index-indj, dims), index_1D(index_terk-indk-indj, dims),
                                    index_1D(index_terk-indj, dims), index_1D(index_terk+indk-indj, dims)]
                        
                        if index[j] < dims[j]-1:
                            data += [m[j,k]*coeff3*c, m[j,k]*coeff3*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]-2]*c,
                                     m[j,k]*coeff3*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]-1]*c,
                                     m[j,k]*coeff3*(-1/dims[k])*ljk[k][index[k]-1,index_terk[k]]*c]
                            row += [i]*4
                            col += [index_1D(index+indj, dims), index_1D(index_terk-indk+indj, dims),
                                    index_1D(index_terk+indj, dims), index_1D(index_terk+indk+indj, dims)]

    return sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()


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
    
    # matrices for drift
    vd = calcD(dims)

    # matrix for selection
    # we don't compute the matrix if not necessary
    if (not s.any()) :
        S1 = [sp.sparse.coo_matrix(([], ([], [])), shape = (dims[i], dims[i]), dtype = 'float').tocsc() for i in range(len(dims))]
    else : S1 = calcS_jk3(dims)
    
    if (not s.any()) or (not (h-0.5).any()):
        S2 = [sp.sparse.coo_matrix(([], ([], [])), shape = (dims[i], dims[i]), dtype = 'float').tocsc() for i in range(len(dims))]
    else : S2 = calcS2_jk3(dims)

    # matrix for migration
    Mi = calcM_jk3(dims, mm)
    
    # time loop:
    sfs = sfs0
    t = 0.0
    
    interval = time.time() - start_time
    print('Time init:', interval)
    start_time = time.time()
    
    while t < Tmax:
        if t+dt>Tmax: dt = Tmax-t
        # we recompute the matrix only if N has changed...
        if (Nold!=N).any():
            D = [1/4.0/N[i]*vd[i] for i in range(len(n))]
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(dims[i], dtype = 'float', format = 'csc')-dt*(D[i]+s[i]*h[i]*S1[i]+s[i]*(1-2.0*h[i])*S2[i])) for i in range(len(n))]
        
        # Backward Euler scheme with splitted operators
        sfs = mutate(sfs, u, dims, dt)
        
        # 1D
        #sfs = slv[0](sfs)
        
        # 2D
        for i in range(int(dims[1])):
            sfs[:,i] = slv[0](sfs[:,i])
        for i in range(int(dims[0])):
            sfs[i,:] = slv[1](sfs[i,:])
        # 3D
        '''
        for i in range(int(dims[1])):
            for j in range(int(dims[2])):
                sfs[:,i,j] = slv[0](sfs[:,i,j])
        for i in range(int(dims[0])):
            for j in range(int(dims[2])):
                sfs[i,:,j] = slv[1](sfs[i,:,j])
        for i in range(int(dims[0])):
            for j in range(int(dims[1])):
                sfs[i,j,:] = slv[2](sfs[i,j,:])'''
        #migrations ???
        sfs = migrate(sfs, Mi, dt)
        
        Nold = N
        t += dt
        N = np.array(fctN(t/(2.0*N0)))
    
    interval = time.time() - start_time
    print('Time loop:', interval)

    return sfs




