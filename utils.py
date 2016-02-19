import numpy as np
import math
import scipy.misc as misc

import integration_multiD as itd
#-----------------------------------------
# Usefull functions for sfs manipulations:
#-----------------------------------------

# split a population into 2 populations
# needs n = n1+n2 (sample sizes)
def split_pop_12(fs,n1,n2):
    assert(len(fs)==n1+n2-1)
    fs2 = np.zeros((n1+1,n2+1))
    for i in range(n1+1):
        for j in range(n2+1):
            if (i+j>0) and (i+j<n1+n2):
                fs2[i,j] = fs[i+j-1]*misc.comb(n1,i)*misc.comb(n2,j)/misc.comb(n1+n2,i+j)
    return fs2

# merge two populations into one population
def merge_pop_21(fs2):
    fs2 = np.array(fs2)
    dim1,dim2 = fs2.shape
    fs = np.zeros(dim1+dim2-3)
    for k in range(dim1):
        for l in range(dim2):
            if k+l>0 and k+l<dim1+dim2-2:
                fs[k+l-1] += fs2[k,l]
    return fs

# project a 1D sfs from sample size m to sample size n (n<=m)
def project_1D(sfs, n):
    m = len(sfs) + 1
    assert(n <= m)
    A = np.zeros((n-1,m-1))
    for i in range(n-1):
        for j in range(i,m-1):
            if (m-j >= n-i):
                A[i,j] = misc.comb(j+1,i+1)*misc.comb(m-j-1,n-i-1)/misc.comb(m,n)
    return np.dot(A, sfs)

# project a 2D sfs from sample size (m1,m2) to sample size (n1, n2)  (n1<=m1, n2<=m2)
def project_2D(sfs, n1, n2):
    m = sfs.shape
    m1 = m[0]-1
    m2 = m[1]-1
    dimm = (m1+1)*(m2+1)
    dimn = (n1+1)*(n2+1)
    sfs1 = sfs.reshape(dimm)
    A = np.zeros((dimn,dimm))
    for i in range(dimn):
        ii = itd.index_nD(i, [n1+1,n2+1])
        for j in range(dimm):
            ij = itd.index_nD(j, [m1+1,m2+1])
            if (ii[0]<=ij[0]) and (ii[1]<=ij[1]) and (n1-ii[0]<=m1-ij[0]) and (n2-ii[1]<=m2-ij[1]):
                A[i,j] = misc.comb(ij[0],ii[0])*misc.comb(m1-ij[0],n1-ii[0])/misc.comb(m1,n1)*misc.comb(ij[1],ii[1])*misc.comb(m2-ij[1],n2-ii[1])/misc.comb(m2,n2)
    sfs1 = np.dot(A,sfs1)
    return sfs1.reshape([n1+1,n2+1])

# Richardson extrapolation
def extrap_lin(sfs1, sfs2, n1, n2):
    assert(sfs1.shape==sfs2.shape)
    return (n2*sfs1-n1*sfs2)/(n2-n1)

def extrap_quad(sfs1, sfs2, sfs3, n1, n2, n3):
    assert(sfs1.shape==sfs2.shape)
    assert(sfs1.shape==sfs3.shape)
    return n2*n3/float((n1-n2)*(n1-n3))*sfs1 + n1*n3/float((n2-n1)*(n2-n3))*sfs2 + n1*n2/float((n3-n1)*(n3-n2))*sfs3
