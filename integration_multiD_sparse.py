import numpy as np
import scipy as sp
from scipy.sparse import linalg
import math

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
        #ind = np.ones(len(dims), dtype='int')
        tp = tuple(ind)
        B[tp] = dims[k]-1
    return u*B

# We compute the  matrices for drift
# this function returns a list of matrices corresponding to each population
# dims -> array containing the dimensions of the problem dims[j] = nj+1
def calcD(dims):
    # number of freedom degrees
    d = int(np.prod(dims))
    # we consider separately the contributions of each dimension
    res = []
    for j in range(len(dims)):
        data = []
        row = []
        col = []
        # creating the ej vector
        ind = np.zeros(len(dims), dtype='int')
        ind[j] = int(1)
        # loop over the fs elements:
        for i in range(0,d):
            # for each element of our nD fs (stored in a vector), we compute its nD index (position in the nD matrix figuring the fs)
            index = index_nD(i, dims)
            # notice that "index[j] = ij"
            if (index[j]>1):
                data.append((index[j]-1)*(dims[j]-index[j]))
                row.append(i)
                col.append(index_1D(index-ind, dims))
            if (index[j]<dims[j]-2):
                data.append((index[j]+1)*(dims[j]-index[j]-2))
                col.append(index_1D(index+ind, dims))
                row.append(i)
            if (index[j]>0) and (index[j]<dims[j]-1):
                data.append(-2*index[j]*(dims[j]-index[j]-1))
                row.append(i)
                col.append(i)
        res.append(sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc())
    return res

# Selection
# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
# with order 3 JK...
def calcS_jk3(dims, s, h):
    # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK13(int(dims[i]-1)))
    d = int(np.prod(dims))

    data = []
    row = []
    col = []

    for i in range(d):
        # multi-D index of the current variable
        index = index_nD(i, dims)
        for j in range(len(dims)):
            ind = np.zeros(len(dims), dtype='int')
            ind[j] = int(1)
            g1 = s[j]*h[j]/dims[j]*index[j]*(dims[j]-index[j])
            g2 = -s[j]*h[j]/dims[j]*(index[j]+1)*(dims[j]-1-index[j])
            index_bis = np.array(index)
            index_bis[j] = jk.index_bis(index_bis[j],dims[j]-1)
            index_ter = np.array(index)+ind
            index_ter[j] = jk.index_bis(index_ter[j],dims[j]-1)
            if (index[j]<dims[j]-1):
                data += [g1*ljk[j][index[j]-1,index_bis[j]-1], g1*ljk[j][index[j]-1,index_bis[j]-2],
                         g1*ljk[j][index[j]-1,index_bis[j]], g2*ljk[j][index[j],index_ter[j]-1],
                         g2*ljk[j][index[j],index_ter[j]-2], g2*ljk[j][index[j],index_ter[j]]]
                row += [i]*6
                col += [index_1D(index_bis, dims), index_1D(index_bis-ind, dims), index_1D(index_bis+ind, dims),
                        index_1D(index_ter, dims), index_1D(index_ter-ind, dims), index_1D(index_ter+ind, dims)]
            
            if index[j]==dims[j]-1: # g2=0
                data += [g1*ljk[j][index[j]-1,index_bis[j]-1], g1*ljk[j][index[j]-1,index_bis[j]-2], g1*ljk[j][index[j]-1,index_bis[j]]]
                row += [i]*3
                col += [index_1D(index_bis, dims), index_1D(index_bis-ind, dims), index_1D(index_bis+ind, dims)]

    return sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()

# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
def calcS2_jk3(dims, s, h):
    # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK23(int(dims[i]-1)))

    d = int(np.prod(dims))

    data = []
    row = []
    col = []
    for i in range(d):
        # multi-D index of the current variable
        index = index_nD(i, dims)
        for j in range(len(dims)):
            ind = np.zeros(len(dims), dtype='int')
            ind[j] = int(1)
            g1 = s[j]*(1-2.0*h[j])*(index[j]+1)/dims[j]/(dims[j]+1)*index[j]*(dims[j]-index[j])
            g2 = -s[j]*(1-2.0*h[j])*(index[j]+1)/dims[j]/(dims[j]+1)*(index[j]+2)*(dims[j]-1-index[j])
            index_ter = np.array(index)+ind
            index_ter[j] = jk.index_bis(index_ter[j],dims[j]-1)
            index_qua = np.array(index)+2*ind
            index_qua[j] = jk.index_bis(index_qua[j],dims[j]-1)
            if index[j]<dims[j]-1:
                data += [g1*ljk[j][index[j],index_ter[j]-1], g1*ljk[j][index[j],index_ter[j]-2],
                         g1*ljk[j][index[j],index_ter[j]], g2*ljk[j][index[j]+1,index_qua[j]-1],
                         g2*ljk[j][index[j]+1,index_qua[j]-2], g2*ljk[j][index[j]+1,index_qua[j]]]
                row += [i]*6
                col += [index_1D(index_ter, dims), index_1D(index_ter-ind, dims), index_1D(index_ter+ind, dims),
                        index_1D(index_qua, dims), index_1D(index_qua-ind, dims), index_1D(index_qua+ind, dims)]
            
            if index[j]==dims[j]-1: # g2=0
                data += [g1*ljk[j][index[j],index_ter[j]-1], g1*ljk[j][index[j],index_ter[j]-2], g1*ljk[j][index[j],index_ter[j]]]
                row += [i]*3
                col += [index_1D(index_ter, dims), index_1D(index_ter-ind, dims), index_1D(index_ter+ind, dims)]

    return sp.sparse.coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()


# Migration
# m -> migration rates matrix, m[i,j] = migration rate from pop i to pop j
# with order 3 JK
def calcM_jk3(dims, m):
    
    if (len(dims)==1) : return  sp.sparse.coo_matrix(([], ([], [])), shape = (dims[0], dims[0]), dtype = 'float').tocsc()
    
    # we precompute the JK3 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK13(int(dims[i]-1)))

    d = int(np.prod(dims))
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

#----------------------------------
# Steady state (for initialization)
#----------------------------------
def steady_state(n, N, gamma, h, m, theta=1.0, reshape=True):
    # parameters of the equation
    mm = np.array(m)/(2.0*N[0])
    s = np.array(gamma)/N[0]
    u = theta/(4.0*N[0])
    # dimensions of the sfs
    dims = n+np.ones(len(n))
    d = int(np.prod(dims))
    
    # matrix for mutations
    B = calcB(u, dims)
    # matrix for drift
    vd = calcD(dims)
    D = 1/4.0/N[0]*vd[0]
    for i in range(1, len(N)):
        D = D + 1/4.0/N[i]*vd[i]
    # matrix for selection
    S = calcS_jk3(dims, s, h)
    S2 = calcS2_jk3(dims, s, h)
    # matrix for migration
    Mi = calcM_jk3(dims, mm)
    Mat = D+S+S2+Mi
    B1 = B.reshape(d)

    sfs = sp.sparse.linalg.spsolve(Mat[1:d-1,1:d-1],-B1[1:d-1])
    sfs = np.insert(sfs, 0, 0.0)
    sfs = np.insert(sfs, d-1, 0.0)
    if reshape: sfs = sfs.reshape(dims)
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

# for a constant N
def integrate_N_cst(sfs0, N, n, tf, dt, gamma, h, m, theta=1.0):
    # parameters of the equation
    mm = np.array(m)/(2.0*N[0])
    s = np.array(gamma)/N[0]
    Tmax = tf*2.0*N[0]
    dt = dt*2.0*N[0]
    u = theta/(4.0*N[0])
    # dimensions of the sfs
    dims = n+np.ones(len(n))
    d = int(np.prod(dims))
    
    # we compute the matrices we will need
    # matrix for mutations
    B = calcB(u, dims)
    # matrix for drift
    vd = calcD(dims)
    D = 1/4.0/N[0]*vd[0]
    for i in range(1, len(N)):
        D = D + 1/4.0/N[i]*vd[i]
    # matrix for selection
    S = calcS_jk3(dims, s, h)
    S2 = calcS2_jk3(dims, s, h)
    # matrix for migration
    Mi = calcM_jk3(dims, mm)

    # system inversion for backward scheme
    Q = sp.sparse.identity(d, dtype = 'float', format = 'csc')-dt*(D+S+S2+Mi)

    # LU decomposition
    solve = linalg.factorized(Q)
    # time loop:
    sfs = sfs0
    t = 0.0
    # all in 1D for the time integration...
    sfs1 = sfs.reshape(d)
    B1 = B.reshape(d)
    while t < Tmax:
        # Backward Euler scheme
        sfs1 = solve(sfs1+dt*B1)
        t += dt
    sfs = sfs1.reshape(dims)
    return sfs

# for a "lambda" definition of N - with Crank Nicholson integration scheme
# fctN is the name of a "lambda" fuction giving N = fctN(t)
# where t is the relative time in generations such as t = 0 initially
# fctN is a lambda function of the time t returning the vector N = (N1,...,Np)
def integrate_N_lambda_CN(sfs0, fctN, n, tf, dt, gamma, h, m, theta=1.0):
    # parameters of the equation
    N = fctN(0)
    N0=N[0]
    mm = np.array(m)/(2.0*N0)
    s = np.array(gamma)/N0
    Tmax = tf*2.0*N0
    dt = dt*2.0*N0
    u = theta/(4.0*N0)
    # dimensions of the sfs
    dims = n+np.ones(len(n))
    d = int(np.prod(dims))
    
    # we compute the matrices we will need
    # matrix for mutations
    B = calcB(u, dims)
    # matrix for drift
    vd = calcD(dims)
    D = 1/4.0/N0*vd[0]
    for i in range(1, len(N)):
        D = D + 1/4.0/N[i]*vd[i]
    # matrix for selection
    S = calcS_jk3(dims, s, h)
    S2 = calcS2_jk3(dims, s, h)
    # matrix for migration
    Mi = calcM_jk3(dims, mm)

    # time loop:
    sfs = sfs0
    t = 0.0
    # all in 1D for the time integration...
    sfs1 = sfs.reshape(d)
    B1 = B.reshape(d)

    while t < Tmax:
        D = 1/4.0/N[0]*vd[0]
        for i in range(1, len(N)):
            D = D + 1/4.0/N[i]*vd[i]
        Q1 = sp.sparse.identity(d, dtype = 'float', format = 'csc')-dt/2*(D+S+S2+Mi)
        Q2 = sp.sparse.identity(d, dtype = 'float', format = 'csc')+dt/2*(D+S+S2+Mi)
        # Crank Nicholson
        sfs1 = sp.sparse.linalg.spsolve(Q1,Q2.dot(sfs1)+dt*B1)
        t += dt
        # we update the populations sizes
        N = fctN(t/(2.0*N0))

    sfs = sfs1.reshape(dims)
    return sfs
