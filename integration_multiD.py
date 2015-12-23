import numpy as np
import math

import jackknife as jk

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
        matd = np.zeros((d,d))
        # creating the ej vector
        ind = np.zeros(len(dims), dtype='int')
        ind[j] = int(1)
        # loop over the fs elements:
        for i in range(0,d):
            # for each element of our nD fs (stored in a vector), we compute its nD index (position in the nD matrix figuring the fs)
            index = index_nD(i, dims)
            # notice that "index[j] = ij"
            if (index[j]>1):
                matd[i,index_1D(index-ind, dims)] = (index[j]-1)*(dims[j]-index[j])
            if (index[j]<dims[j]-2):
                matd[i,index_1D(index+ind, dims)] = (index[j]+1)*(dims[j]-index[j]-2)
            matd[i,i] = -2*index[j]*(dims[j]-index[j]-1)
        res.append(matd)
    return res

# Selection
# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
# this function includes JK2 for higher order terms estimation
def calcS(dims, s, h):
    # we precompute the JK2 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK12(int(dims[i]-1)))
    
    d = int(np.prod(dims))
    S = np.zeros((d,d))
    for i in range(d):
        # multi-D index of the current variable
        index = index_nD(i, dims)
        for j in range(len(dims)):
            ind = np.zeros(len(dims), dtype='int')
            ind[j] = int(1)
            g1 = s[j]*h[j]/(dims[j])*index[j]*(dims[j]-index[j])
            g2 = s[j]*h[j]/(dims[j])*(index[j]+1)*(dims[j]-1-index[j])
            if (index[j]>1) and (index[j]<dims[j]-2):
                S[i,i] += g1*ljk[j][index[j]-1,index[j]-1]-g2*ljk[j][index[j],index[j]-1]
                S[i,index_1D(index-ind, dims)] += g1*ljk[j][index[j]-1,index[j]-2]
                S[i,index_1D(index+ind, dims)] -= g2*ljk[j][index[j],index[j]]
            
            if index[j]==1:
                S[i,i] += g1*ljk[j][0,0]-g2*ljk[j][1,0]
                S[i,index_1D(index+ind, dims)] += g1*ljk[j][0,1]-g2*ljk[j][1,1]
                    
            if index[j]==dims[j]-2:
                S[i,i] += g1*ljk[j][dims[j]-3,dims[j]-3]-g2*ljk[j][dims[j]-2,dims[j]-3]
                S[i,index_1D(index-ind, dims)] += g1*ljk[j][dims[j]-3,dims[j]-4]-g2*ljk[j][dims[j]-2,dims[j]-4]

            if index[j]==0: # g1=0
                S[i,index_1D(index+ind, dims)] -= g2*ljk[j][0,0]
                S[i,index_1D(index+2*ind, dims)] -= g2*ljk[j][0,1]

            if index[j]==dims[j]-1: # g2=0
                S[i,index_1D(index-ind, dims)] += g1*ljk[j][dims[j]-2,dims[j]-3]
                S[i,index_1D(index-2*ind, dims)] += g1*ljk[j][dims[j]-2,dims[j]-4]

    return S

# s -> array containing the selection coefficients for each population [s1, s2, ..., sp]
# h -> [h1, h2, ..., hp]
# this function includes JK2 for higher order terms estimation
def calcS2(dims, s, h):
    # we precompute the JK2 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK12(int(dims[i]-1)))
    ljk2 = [] # for h != 1/2
    for i in range(len(dims)):
        ljk2.append(np.dot(jk.calcJK12(int(dims[i])),ljk[i]))

    d = int(np.prod(dims))
    S = np.zeros((d,d))
    for i in range(d):
        # multi-D index of the current variable
        index = index_nD(i, dims)
        for j in range(len(dims)):
            ind = np.zeros(len(dims), dtype='int')
            ind[j] = int(1)
            g1 = s[j]*(1-2.0*h[j])*(index[j]+1)/dims[j]/(dims[j]+1)*index[j]*(dims[j]-index[j])
            g2 = s[j]*(1-2.0*h[j])*(index[j]+1)/dims[j]/(dims[j]+1)*(index[j]+2)*(dims[j]-1-index[j])
            if (index[j]>1) and (index[j]<dims[j]-3):
                S[i,i] += g1*ljk2[j][index[j],index[j]-1]-g2*ljk2[j][index[j]+1,index[j]-1]
                S[i,index_1D(index-ind, dims)] += g1*ljk2[j][index[j],index[j]-2]
                S[i,index_1D(index+ind, dims)] += g1*ljk2[j][index[j],index[j]]-g2*ljk2[j][index[j]+1,index[j]]
                S[i,index_1D(index+2*ind, dims)] -= g2*ljk2[j][index[j]+1,index[j]+1]
            
            if index[j]==0: # g1=0
                S[i,index_1D(index+ind, dims)] -= g2*ljk2[j][1,0]
                S[i,index_1D(index+2*ind, dims)] -= g2*ljk2[j][1,1]
            
            if index[j]==1:
                S[i,i] += g1*ljk2[j][1,0]-g2*ljk2[j][2,0]
                S[i,index_1D(index+ind, dims)] += g1*ljk2[j][1,1]-g2*ljk2[j][2,1]
                S[i,index_1D(index+2*ind, dims)] += -g2*ljk2[j][2,2]
            if index[j]==dims[j]-3:
                S[i,i] += g1*ljk2[j][dims[j]-3,dims[j]-4]-g2*ljk2[j][dims[j]-2,dims[j]-4]
                S[i,index_1D(index+ind, dims)] += g1*ljk2[j][dims[j]-3,dims[j]-3]-g2*ljk2[j][dims[j]-2,dims[j]-3]
                S[i,index_1D(index-ind, dims)] += g1*ljk2[j][dims[j]-3,dims[j]-5]
            
            if index[j]==dims[j]-2:
                S[i,i] += g1*ljk2[j][dims[j]-2,dims[j]-3]-g2*ljk2[j][dims[j]-1,dims[j]-3]
                S[i,index_1D(index-ind, dims)] += g1*ljk2[j][dims[j]-2,dims[j]-4]-g2*ljk2[j][dims[j]-1,dims[j]-4]
            
            if index[j]==dims[j]-1: # g2=0
                S[i,index_1D(index-ind, dims)] += g1*ljk2[j][dims[j]-1,dims[j]-3]
                S[i,index_1D(index-2*ind, dims)] += g1*ljk2[j][dims[j]-1,dims[j]-4]

    return S

# Migration
# m -> migration rates matrix, m[i,j] = migration rate from pop i to pop j
def calcM(dims, m):
    # just if we have at least 2 populations...
    assert(len(dims>1))
    # we precompute the JK2 coefficients we will need (same as in 1D)...
    ljk = []
    for i in range(len(dims)):
        ljk.append(jk.calcJK12(int(dims[i]-1)))

    d = int(np.prod(dims))
    M = np.zeros((d,d))
    for i in range(d):
        # multi-D index of the current variable
        index = index_nD(i, dims)
        for j in range(len(dims)):
            indj = np.zeros(len(dims), dtype='int')
            indj[j] = int(1)
            coeff1 = (2*index[j]-(dims[j]-1))/(dims[j]-1)
            coeff2 = (dims[j]-index[j])/(dims[j]-1)
            coeff3 = (index[j]+1)/(dims[j]-1)
            for k in range(len(dims)):
                if k!=j:
                    indk = np.zeros(len(dims), dtype='int')
                    indk[k] = int(1)
                    
                    M[i,i] -= m[j,k]*index[j]
                    if index[j]>0:
                        M[i,index_1D(index-indj, dims)] += m[j,k]*(index[j]+1)
                    
                    if index[k]>0 and index[k]<dims[k]-2:
                        M[i,i] += m[j,k]*coeff1*ljk[k][index[k],index[k]-1]
                        M[i,index_1D(index+indk, dims)] += m[j,k]*coeff1*ljk[k][index[k],index[k]]
                        if index[j] > 0:
                            M[i,index_1D(index+indk-indj, dims)] += m[j,k]*coeff2*ljk[k][index[k],index[k]]
                            M[i,index_1D(index-indj, dims)] += m[j,k]*coeff2*ljk[k][index[k],index[k]-1]
                        if index[j] < dims[j]-1:
                            M[i,index_1D(index+indk+indj, dims)] -= m[j,k]*coeff3*ljk[k][index[k],index[k]]
                            M[i,index_1D(index+indj, dims)] -= m[j,k]*coeff3*ljk[k][index[k],index[k]-1]
                    if index[k]==0:
                        M[i,index_1D(index+indk, dims)] += m[j,k]*coeff1*ljk[k][0,0]
                        M[i,index_1D(index+2*indk, dims)] += m[j,k]*coeff1*ljk[k][0,1]
                        if index[j] > 0:
                            M[i,index_1D(index+indk-indj, dims)] += m[j,k]*coeff2*ljk[k][0,0]
                            M[i,index_1D(index+2*indk-indj, dims)] += m[j,k]*coeff2*ljk[k][0,1]
                        if index[j] < dims[j]-1:
                            M[i,index_1D(index+indk+indj, dims)] -= m[j,k]*coeff3*ljk[k][0,0]
                            M[i,index_1D(index+2*indk+indj, dims)] -= m[j,k]*coeff3*ljk[k][0,1]
                    if index[k]==dims[k]-2:
                        M[i,i] += m[j,k]*coeff1*ljk[k][dims[k]-3,dims[k]-3]
                        M[i,index_1D(index-indk, dims)] += m[j,k]*coeff1*ljk[k][dims[k]-3,dims[k]-4]
                        if index[j] > 0:
                            M[i,index_1D(index-indj, dims)] += m[j,k]*coeff2*ljk[k][dims[k]-3,dims[k]-3]
                            M[i,index_1D(index-indk-indj, dims)] += m[j,k]*coeff2*ljk[k][dims[k]-3,dims[k]-4]
                        if index[j] < dims[j]-1:
                            M[i,index_1D(index+indj, dims)] += m[j,k]*coeff3*ljk[k][dims[k]-3,dims[k]-3]
                            M[i,index_1D(index-indk+indj, dims)] += m[j,k]*coeff3*ljk[k][dims[k]-3,dims[k]-4]
                    if index[k]==dims[k]-1:
                        M[i,i] += m[j,k]*coeff1
                        M[i,index_1D(index-indk, dims)] += m[j,k]*coeff1*(-1/dims[k])*ljk[k][dims[k]-2,dims[k]-3]
                        M[i,index_1D(index-2*indk, dims)] += m[j,k]*coeff1*(-1/dims[k])*ljk[k][dims[k]-2,dims[k]-4]
                        if index[j] > 0:
                            M[i,index_1D(index-indj, dims)] += m[j,k]*coeff2
                            M[i,index_1D(index-indk-indj, dims)] += m[j,k]*coeff2*(-1/dims[k])*ljk[k][dims[k]-2,dims[k]-3]
                            M[i,index_1D(index-2*indk-indj, dims)] += m[j,k]*coeff2*(-1/dims[k])*ljk[k][dims[k]-2,dims[k]-4]
                        if index[j] < dims[j]-1:
                            M[i,index_1D(index+indj, dims)] += m[j,k]*coeff3
                            M[i,index_1D(index-indk+indj, dims)] += m[j,k]*coeff3*(-1/dims[k])*ljk[k][dims[k]-2,dims[k]-3]
                            M[i,index_1D(index-2*indk+indj, dims)] += m[j,k]*coeff3*(-1/dims[k])*ljk[k][dims[k]-2,dims[k]-4]


    return M















