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
def migrate(sfs, slv, dt):
    dims = sfs.shape
    d = int(np.prod(dims))
    #nd = len(dims)
    fs = sfs.copy()
    fs = fs.reshape(d)
    fs = slv(fs)
    fs = fs.reshape(dims)
    return fs

# Migration
# m -> migration rates matrix, m[i,j] = migration rate from pop i to pop j
# with order 3 JK
def calcM1_jk3(dims, m):
    # number of degrees of freedom
    d = int(np.prod(dims))
    
    # we don't compute the matrix if not necessary
    if (len(dims)==1) : return  [sp.sparse.coo_matrix(([], ([], [])), shape = (dims[0], dims[0]), dtype = 'float').tocsc()]
    if (not m.any()) : return  [sp.sparse.coo_matrix(([], ([], [])), shape = (dims[i], dims[i]), dtype = 'float').tocsc() for i in range(len(dims))]
    
    # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK13(int(dims[i]-1)))
    
    res = []
    for j in range(len(dims)):
        
        data = []
        row = []
        col = []
        
        # total migration rate from pop j
        mj = np.sum(m[j,:])-m[j,j]

        for i in range(int(dims[j])):
            
            data.append(-mj*i)
            row.append(i)
            col.append(i)
            
            if i < dims[j]-1:
                data.append(mj*(i+1))
                row.append(i)
                col.append(i+1)
        res.append(sp.sparse.coo_matrix((data, (row, col)), shape = (dims[j], dims[j]), dtype = 'float').tocsc())
    return res


def calcM_jk3bis(dims,m):
    res = []
    for i in range(len(dims)):
        for j in range(i+1, len(dims)):
            mbis = np.array([[0,m[i,j]],[m[j,i],0]])
            res.append(its.calcM_jk3([dims[i],dims[j]],mbis))
    #print(i,j)
    return res
# Migration
# m -> migration rates matrix, m[i,j] = migration rate from pop i to pop j
# with order 3 JK
def calcM_jk3(dims, m):
    # we don't compute the matrix if there is just 1 pop
    if (len(dims)==1) : return  sp.sparse.coo_matrix(([], ([], [])), shape = (dims[0], dims[0]), dtype = 'float').tocsc()
   
   # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK13(int(dims[i]-1)))

    res = []

    for j in range(len(dims)):
        res_dim = []
        
        #indj = np.zeros(len(dims), dtype='int')
        #indj[j] = int(1)
        indj = np.array([1, 0])
        for k in range(len(dims)):
            d = int(dims[j]*dims[k])
            if (m[j,k]==0):
                res_dim.append(sp.sparse.coo_matrix(([], ([], [])), shape = (d, d), dtype = 'float').tocsc())
                break
            if k != j:
                
                data = []
                row = []
                col = []
                for i in range(d):
        
                    # 2D index of the current variable
                    index = index_nD(i, [dims[j], dims[k]])
                    #print('index', index, k)
                    '''index_bisj = np.array(index)
                    index_bisj[j] = jk.index_bis(index_bisj[j],dims[j]-1)
                    index_terj = np.array(index)+indj
                    index_terj[j] = jk.index_bis(index_terj[j],dims[j]-1)'''
        
                    coeff1 = m[j,k]*(2*index[0]-(dims[j]-1))*(index[1]+1)/dims[k]
                    coeff2 = m[j,k]*(dims[j]-index[0])*(index[1]+1)/dims[k]
                    coeff3 = -m[j,k]*(index[0]+1)*(index[1]+1)/dims[k]
            
                    indk = np.array([0, 1])#np.zeros(len(dims), dtype='int')
                    #indk[k] = int(1)
                    
                    
                    index_bisk = np.array(index)
                    index_bisk[1] = jk.index_bis(index_bisk[1],dims[k]-1)
                    index_terk = np.array(index)+indk
                    index_terk[1] = jk.index_bis(index_terk[1],dims[k]-1)
                    
                    if index[1] < dims[k]-1:
                        data += [coeff1*ljk[k][index[1],index_terk[1]-2], coeff1*ljk[k][index[1],index_terk[1]-1], coeff1*ljk[k][index[1],index_terk[1]]]
                        row += [i]*3
                        col += [index_1D(index_terk-indk, [dims[j], dims[k]]), index_1D(index_terk, [dims[j], dims[k]]), index_1D(index_terk+indk, [dims[j], dims[k]])]
                        if index[0] > 0:
                            data +=[coeff2*ljk[k][index[1],index_terk[1]-2], coeff2*ljk[k][index[1],index_terk[1]-1], coeff2*ljk[k][index[1],index_terk[1]]]
                            row += [i]*3
                            col += [index_1D(index_terk-indk-indj, [dims[j], dims[k]]), index_1D(index_terk-indj, [dims[j], dims[k]]), index_1D(index_terk+indk-indj, [dims[j], dims[k]])]
                        if index[0] < dims[j]-1:
                            data += [coeff3*ljk[k][index[1],index_terk[1]-2], coeff3*ljk[k][index[1],index_terk[1]-1], coeff3*ljk[k][index[1],index_terk[1]]]
                            row += [i]*3
                            col += [index_1D(index_terk-indk+indj, [dims[j], dims[k]]), index_1D(index_terk+indj, [dims[j], dims[k]]), index_1D(index_terk+indk+indj, [dims[j], dims[k]])]
                            
                    if index[1] == dims[k]-1:
                        data += [coeff1, coeff1*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]-2],
                                 coeff1*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]-1],
                                 coeff1*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]]]
                        row += [i]*4
                        col += [i, index_1D(index_terk-indk, [dims[j], dims[k]]), index_1D(index_terk, [dims[j], dims[k]]), index_1D(index_terk+indk, [dims[j], dims[k]])]

                        if index[0] > 0:
                            data += [coeff2, coeff2*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]-2],
                                     coeff2*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]-1],
                                     coeff2*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]]]
                            row += [i]*4
                            col += [index_1D(index-indj, [dims[j], dims[k]]), index_1D(index_terk-indk-indj, [dims[j], dims[k]]),
                                    index_1D(index_terk-indj, [dims[j], dims[k]]), index_1D(index_terk+indk-indj, [dims[j], dims[k]])]
                        
                        if index[0] < dims[j]-1:
                            data += [coeff3, coeff3*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]-2],
                                     coeff3*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]-1],
                                     coeff3*(-1/dims[k])*ljk[k][index[1]-1,index_terk[1]]]
                            row += [i]*4
                            col += [index_1D(index+indj, [dims[j], dims[k]]), index_1D(index_terk-indk+indj, [dims[j], dims[k]]),
                                    index_1D(index_terk+indj, [dims[j], dims[k]]), index_1D(index_terk+indk+indj, [dims[j], dims[k]])]
                        res_dim.append(sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc())
                res.append(res_dim)
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

    Mi = calcM_jk3bis(dims,mm)

    #mimi = its.calcM_jk3(dims,mm)
    #slvm = linalg.factorized(sp.sparse.identity(mimi.shape[0], dtype = 'float', format = 'csc')-dt*mimi)
    # solvers for migration
    slvm = [linalg.factorized(sp.sparse.identity(Mi[i].shape[0], dtype = 'float', format = 'csc')-1*dt*Mi[i]) for i in range(len(Mi))]

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
            D = [1/4.0/N[i]*vd[i] for i in range(len(n))]
            # system inversion for backward scheme
            slv = [linalg.factorized(sp.sparse.identity(dims[i], dtype = 'float', format = 'csc')-dt*(D[i]+s[i]*h[i]*S1[i]+s[i]*(1-2.0*h[i])*S2[i])) for i in range(len(n))]

        
        # Backward Euler scheme with splitted operators
        sfs = mutate(sfs, u, dims, dt)
        
        # 1D
        if len(dims)==1: sfs = slv[0](sfs)
        
        # 2D
        if len(dims)==2:
            #sfstemp = sfs.reshape(dims[0]*dims[1])
            #sfs = (slvm[0](sfstemp)).reshape(dims)

            for i in range(int(dims[1])):
                sfs[:,i] = slv[0](sfs[:,i])
            for i in range(int(dims[0])):
                sfs[i,:] = slv[1](sfs[i,:])
            # migrations
            sfstemp = sfs.reshape(dims[0]*dims[1])
            sfs = (slvm[0](sfstemp)).reshape(dims)


        # 3D
        if len(dims)==3:
            for i in range(int(dims[1])):
                for j in range(int(dims[2])):
                    sfs[:,i,j] = slv[0](sfs[:,i,j])
            for i in range(int(dims[0])):
                for j in range(int(dims[2])):
                    sfs[i,:,j] = slv[1](sfs[i,:,j])
            for i in range(int(dims[0])):
                for j in range(int(dims[1])):
                    sfs[i,j,:] = slv[2](sfs[i,j,:])
            # migrations
            for i in range(int(dims[0])):
                sfs[i,:,:] = slvm[2](sfs[i,:,:].reshape(dims[1]*dims[2])).reshape(dims[1],dims[2])
            for i in range(int(dims[1])):
                sfs[:,i,:] = slvm[1](sfs[:,i,:].reshape(dims[0]*dims[2])).reshape(dims[0],dims[2])
            for i in range(int(dims[2])):
                sfs[:,:,i] = slvm[0](sfs[:,:,i].reshape(dims[0]*dims[1])).reshape(dims[0],dims[1])
        Nold = N
        t += dt
        N = np.array(fctN(t/(2.0*N0)))
    
    interval = time.time() - start_time
    print('Time loop:', interval)

    return sfs




