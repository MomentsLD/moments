import numpy as np
cimport numpy as np
from scipy.sparse import linalg, coo_matrix

import Jackknife as jk

# needed for finite genome steady state spectrum calculation
import scipy.special as scisp
from mpmath import hyp1f1,mp
mp.dps = 25; mp.pretty = True

"""
Functions to build the matrices we need for the linear system
The code below is written for 1D cases.
"""

"""
Matrix for mutations (forward and backward)
dims = n1+1
"""
cpdef calcB_FB(int d, np.float64_t u, np.float64_t v):
    cdef int i
    cdef list data, row, col
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        if i > 0:
            data += [u * (d-i), -v * i]
            row += 2 * [i]
            col += [i - 1, i]
        if i < d - 1:
            data += [-u * (d-i-1), v * (i+1)]
            row += 2 * [i]
            col += [i, i + 1]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()


"""
Matrix for drift
dims = n1+1
"""
cpdef calcD(int d):
    cdef int i
    cdef list data, row, col
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        if i > 1:
            data.append((i-1) * (d-i))
            row.append(i)
            col.append(i - 1)
        if i < d - 2:
            data.append((i+1) * (d-i-2))
            col.append(i + 1)
            row.append(i)
        if i > 0 and i < d - 1:
            data.append(-2 * i * (d-i-1))
            row.append(i)
            col.append(i)

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

cpdef calcD_dense(int d):
    cdef int i
    cdef np.ndarray[np.float64_t, ndim = 2] res = np.zeros([d, d])
    # loop over the fs elements:
    for i in range(d):
        if i > 1:
            res[i, i - 1] = (i-1) * (d-i)
        if i < d - 2:
            res[i, i + 1] = (i+1) * (d-i-2)
        if i > 0 and i < d - 1:
            res[i, i] = -2 * i * (d-i-1)
    return res


"""
Matrices for selection with order 3 JK
dims = n1+1
ljk is the Jacknife array corresponding to the concerned population size,
"""
# selection with h = 0.5
cpdef calcS(int d, np.ndarray[np.float64_t, ndim = 2] ljk):
    # Computes the jackknife-transformed selection matrix 1
    # for the addition of a single sample
    cdef int i, i_bis, i_ter
    cdef list data, row, col
    cdef np.float64_t g1, g2
    # arrays for the creation of the sparse (coo) matrix
    # data will have matrix entry, row + column have coordinates
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        i_bis = jk.index_bis(i, d - 1) # This picks the second jackknife index 
        i_ter = jk.index_bis(i + 1, d - 1) # This picks the third jackknife index
        # coefficients of the selection matrix
        g1 = i * (d-i) / np.float64(d)
        g2 = -(i+1) * (d-1-i) / np.float64(d)

        if i < d - 1 and i > 0: # First deal with non-fixed variants
            data += [g1 * ljk[i - 1, i_bis - 1], g1 * ljk[i - 1, i_bis - 2],
                    g1 * ljk[i - 1, i_bis], g2 * ljk[i, i_ter - 1],
                    g2 * ljk[i, i_ter - 2], g2 * ljk[i, i_ter]]
            row += 6 * [i]
            col += [i_bis, i_bis - 1, i_bis + 1,
                    i_ter, i_ter - 1, i_ter + 1]
        
        elif i == 0: # g1=0
            data += [g2 * ljk[i, i_ter - 1],
                     g2 * ljk[i, i_ter - 2], g2 * ljk[i, i_ter]]
            row += 3 * [i]
            col += [i_ter, i_ter - 1, i_ter + 1]
        
        elif i == d - 1: # g2=0
            data += [g1 * ljk[i - 1, i_bis - 1], g1 * ljk[i - 1, i_bis - 2],
                     g1 * ljk[i - 1, i_bis]]
            row += 3 * [i]
            col += [i_bis, i_bis - 1, i_bis + 1]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()



# selection with h != 0.5
cpdef calcS2(int d, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int i, i_qua, i_ter
    cdef list data, row, col
    cdef np.float64_t g1, g2
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for i in range(d):
        i_ter = jk.index_bis(i + 1, d - 1)
        i_qua = jk.index_bis(i + 2, d - 1)
        # coefficients
        g1 = (i+1) / np.float64(d) / (d+1.0) * i * (d-i)
        g2 = -(i+1) / np.float64(d) / (d+1.0) * (i+2) * (d-1-i)
        
        if i < d - 1:
            data += [g1 * ljk[i, i_ter - 1], g1 * ljk[i, i_ter - 2],
                     g1 * ljk[i, i_ter], g2 * ljk[i + 1, i_qua - 1],
                     g2 * ljk[i + 1, i_qua - 2], g2 * ljk[i + 1, i_qua]]
            row += 6 * [i]
            col += [i_ter, i_ter - 1, i_ter + 1,
                    i_qua, i_qua - 1, i_qua + 1]
    
        elif i == d - 1: # g2=0
            data += [g1 * ljk[i, i_ter - 1], g1 * ljk[i, i_ter - 2], g1 * ljk[i, i_ter]]
            row += 3 * [i]
            col += [i_ter, i_ter - 1, i_ter + 1]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()



# under/over-dominance
cpdef calcUnderdominance(int d, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int i, n, i0, i1, i2
    cdef list data1, row1, col1
    cdef list data2, row2, col2
    cdef np.float64_t g1, g2
    n = d - 1
    data = []
    row = []
    col = []

    for i in range(d):
        i0 = jk.index_bis(i, n)
        i1 = jk.index_bis(i + 1, n)
        i2 = jk.index_bis(i + 2, n)

        # first term
        fac = 1. / (n + 2.) / (n + 1.)
        g1 = fac * i * (n - i + 2) * (n - i + 1)
        g2 = -fac * (i + 1) * (n - i + 1) * (n - i)
        if i > 0 and i < n:
            data += [
                g1 * ljk[i - 1, i0 - 1], g1 * ljk[i - 1, i0 - 2], g1 * ljk[i - 1, i0],
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 6 * [i]
            col += [i0, i0 - 1, i0 + 1,
                     i1, i1 - 1, i1 + 1]
        elif i == 0:
            data += [
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 3 * [i]
            col += [i1, i1 - 1, i1 + 1]
        elif i == n:
            data += [
                g1 * ljk[i - 1, i0 - 1], g1 * ljk[i - 1, i0 - 2], g1 * ljk[i - 1, i0]
            ]
            row += 3 * [i]
            col += [i0, i0 - 1, i0 + 1]

        # second term
        g1 = fac * (i + 2) * (i + 1) * (n - i)
        g2 = -fac * (i + 1) * i * (n - i + 1)
        if i > 0 and i < n:
            data += [
                g1 * ljk[i + 1, i2 - 1], g1 * ljk[i + 1, i2 - 2], g1 * ljk[i + 1, i2],
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 6 * [i]
            col += [i2, i2 - 1, i2 + 1,
                     i1, i1 - 1, i1 + 1]
        elif i == n:
            data += [
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 3 * [i]
            col += [i1, i1 - 1, i1 + 1]
        elif i == 0:
            data += [
                g1 * ljk[i + 1, i2 - 1], g1 * ljk[i + 1, i2 - 2], g1 * ljk[i + 1, i2]
            ]
            row += 3 * [i]
            col += [i2, i2 - 1, i2 + 1]
    
    M = coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()
    return M


"""
Steady state for 1D population
"""
cpdef np.ndarray[np.float64_t] steady_state_1D(int n, float N=1.0, float gamma=0.0,
                                               float h=0.5, float theta=1.0,
                                               float overdominance=0.0):
    # Update ModelPlot if necessary
    import moments.ModelPlot as ModelPlot
    model = ModelPlot._get_model()
    if model is not None:
        model.initialize(1)
    
    cdef int d
    # dimensions of the sfs
    d = n + 1
    
    # matrix for mutations
    u = np.float64(theta / 4.0)
    s = np.float64(gamma)

    B = np.zeros(d)
    B[1] = n * u

    # matrix for drift
    D = 1 / 4.0 / N * calcD(d)
    # jackknife matrices
    ljk = jk.calcJK13(int(d - 1))
    ljk2 = jk.calcJK23(int(d - 1))
    # matrix for selection
    S = s * h * calcS(d, ljk)

    S2 = s * (1-2.0*h) * calcS2(d, ljk2)

    S3 = h * overdominance * calcUnderdominance(d, ljk2)

    # matrix for migration
    Mat = D + S + S2 + S3

    sfs = linalg.spsolve(Mat[1:d-1, 1:d-1], -B[1:d-1])
    sfs = np.insert(sfs, 0, 0.0)
    sfs = np.insert(sfs, d - 1, 0.0)

    return sfs

"""
Steady state solution for 1D population with reversible mutation
We require h=1/2
These are found by integrating equations 5.70 and 5.72 in Ewens (2004)
        against the binomial sampling function
"""
cpdef np.ndarray[np.float64_t] steady_state_1D_reversible(
    int n,
    float gamma=0.0,
    float theta_fd=0.0008,
    float theta_bd=0.0008
):
    fs = np.zeros(n+1)
    if gamma == 0.0:
        for i in range(n+1):
            fs[i] = scisp.gammaln(n+1) - scisp.gammaln(n-i+1) - scisp.gammaln(i+1) + scisp.gammaln(i+theta_fd) + scisp.gammaln(n-i+theta_bd)
        fs += scisp.gammaln(theta_fd+theta_bd) - scisp.gammaln(theta_fd) - scisp.gammaln(theta_bd) - scisp.gammaln(n+theta_fd+theta_bd)
        fs = np.exp(fs)
    else:
        ## unstable for large n
        for i in range(n+1):
            fs[i] = np.exp(scisp.gammaln(n+1) - scisp.gammaln(n-i+1) - scisp.gammaln(i+1) + scisp.gammaln(i+theta_fd) + scisp.gammaln(n-i+theta_bd) - scisp.gammaln(n+theta_fd+theta_bd)) * hyp1f1(i+theta_fd,n+theta_fd+theta_bd,2*gamma)
    return fs/np.sum(fs)
