import numpy as np
cimport numpy as np
from scipy.sparse import linalg, coo_matrix

import Jackknife as jk

"""
Functions to build the matrices we need for the linear system
The code below is written for 1D cases.
"""

"""
Matrix for drift
dims = n1+1
"""
cpdef calcD(int d):
    cdef int i
    cdef list data,row, col
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        if (i > 1):
            data.append((i-1) * (d-i))
            row.append(i)
            col.append(i-1)
        if (i < d-2):
            data.append((i+1) * (d-i-2))
            col.append(i+1)
            row.append(i)
        if (i > 0) and (i < d-1):
            data.append(-2 * i * (d-i-1))
            row.append(i)
            col.append(i)

    return coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()

"""
Matrices for selection with order 3 JK
dims = n1+1
ljk is the Jacknife array corresponding to the concerned population size,
"""
# selection with h = 0.5
cpdef calcS(int d, np.ndarray ljk):
    cdef int i, i_bis, i_ter
    cdef list data, row, col
    cdef np.float64_t g1, g2
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        i_bis = jk.index_bis(i, d-1)
        i_ter = jk.index_bis(i+1, d-1)
        # coefficients
        g1 = i * (d-i) / np.float64(d)
        g2 = -(i+1) * (d-1-i) / np.float64(d)

        if (i < d-1) and (i > 0):
            data += [g1 * ljk[i-1, i_bis-1], g1 * ljk[i-1, i_bis-2],
                    g1 * ljk[i-1, i_bis], g2 * ljk[i, i_ter-1],
                    g2 * ljk[i, i_ter-2], g2 * ljk[i, i_ter]]
            row += [i, i, i, i, i, i]
            col += [i_bis, i_bis-1, i_bis+1,
                    i_ter, i_ter-1, i_ter+1]
        
        elif i == 0: # g1=0
            data += [g2 * ljk[i, i_ter-1],
                     g2 * ljk[i, i_ter-2], g2 * ljk[i, i_ter]]
            row += [i, i, i]
            col += [i_ter, i_ter-1, i_ter+1]
        
        elif i == d-1: # g2=0
            data += [g1 * ljk[i-1, i_bis-1], g1 * ljk[i-1, i_bis-2], g1 * ljk[i-1, i_bis]]
            row += [i, i, i]
            col += [i_bis, i_bis-1, i_bis+1]

    return coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()

'''
cpdef calcS_bis(int d, np.ndarray ljk):
    cdef int i,
    cdef list data, row, col
    cdef np.float64_t g1, g2
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        # coefficients
        g1 = i * (d-i) / np.float64(d)
        g2 = -(i+1) * (d-1-i) / np.float64(d)
        
        if (i < d-3) and (i > 1):
            data += [g1 * ljk[i-1, i-2],
                     g1*ljk[i-1, i-1] + g2*ljk[i, i-1],
                     g1*ljk[i-1, i] + g2*ljk[i, i],
                     g2 * ljk[i, i+1]]
            row += 4 * [i]
            col += [i-1, i, i+1, i+2]
        
        elif i == 0: # g1=0
            data += [g2 * ljk[0, 0], g2 * ljk[0, 1]]
            row += 2 * [0]
            col += [1, 2]
        
        elif i == 1:
            data += [g1*ljk[0, 0] + g2*ljk[1, 0],
                     g1*ljk[0, 1] + g2*ljk[1, 1],
                     g2 * ljk[1, 2]]
            row += 3 * [1]
            col += [1, 2, 3]
        
        elif i == d-3:
            data += [g1*ljk[d-4, d-5],
                     g1*ljk[d-4, d-4] + g2*ljk[d-3, d-4],
                     g1*ljk[d-4, d-3] + g2*ljk[d-3, d-3]]
            row += 3 * [d-3]
            col += [d-4, d-3, d-2]
        
        elif i == d-2:
            data += [g1*ljk[d-3, d-4] + g2*ljk[d-2, d-4],
                     g1*ljk[d-3, d-3] + g2*ljk[d-2, d-3]]
            row += 2 * [d-2]
            col += [d-3, d-2]
        
        elif i == d-1: # g2=0
            data += [g1 * ljk[d-2, d-4], g1 * ljk[d-2, d-3]]
            row += 2 * [d-1]
            col += [d-3, d-2]

    return coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()'''

# selection with h != 0.5
cpdef calcS2(int d, np.ndarray ljk):
    cdef int i, i_qua, i_ter
    cdef list data, row, col
    cdef np.float64_t g1, g2
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for i in range(d):
        i_ter = jk.index_bis(i+1, d-1)
        i_qua = jk.index_bis(i+2, d-1)
        # coefficients
        g1 = (i+1) / np.float64(d) / (d+1.0) * i * (d-i)
        g2 = -(i+1) / np.float64(d) / (d+1.0) * (i+2) * (d-1-i)
        
        if i < d-1:
            data += [g1 * ljk[i, i_ter-1], g1 * ljk[i, i_ter-2],
                     g1 * ljk[i, i_ter], g2 * ljk[i+1, i_qua-1],
                     g2 * ljk[i+1, i_qua-2], g2 * ljk[i+1, i_qua]]
            row += 6 * [i]
            col += [i_ter, i_ter-1, i_ter+1,
                    i_qua, i_qua-1, i_qua+1]
    
        elif i == d-1: # g2=0
            data += [g1 * ljk[i, i_ter-1], g1 * ljk[i, i_ter-2], g1 * ljk[i, i_ter]]
            row += 3 * [i]
            col += [i_ter, i_ter-1, i_ter+1]

    return coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()

'''
cpdef calcS2_bis(int d, np.ndarray ljk):
    cdef int i, i_qua, i_ter
    cdef list data, row, col
    cdef np.float64_t g1, g2
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for i in range(d):
        i_ter = i+1#jk.index_bis(i+1, d-1)
        i_qua = i+2#jk.index_bis(i+2, d-1)
        # coefficients
        g1 = (i+1) / np.float64(d) / (d+1.0) * i * (d-i)
        g2 = -(i+1) / np.float64(d) / (d+1.0) * (i+2) * (d-1-i)
        
        if (i < d-4) and (i > 0):
            data += [g1 * ljk[i, i-1],
                     g1 * ljk[i, i] + g2 * ljk[i+1, i],
                     g1*ljk[i, i+1] + g2*ljk[i+1, i+1],
                     g2 * ljk[i+1, i+2]]
            row += 4 * [i]
            col += [i, i+1, i+2, i+3]
        
        elif i == 0:
            data += [g2 * ljk[1, 0], g2 * ljk[1, 1], g2 * ljk[1, 2]]
            row += 3 * [0]
            col += [1, 2, 3]
        
        elif i == d-4:
            data += [g1 * ljk[d-4, d-5],
                     g1*ljk[d-4, d-4] + g2*ljk[d-3, d-4],
                     g1*ljk[d-4, d-3] + g2*ljk[d-3, d-3]]
            row += 3 * [d-4]
            col += [d-4, d-3, d-2]
        
        elif i == d-3:
            data += [g1*ljk[d-3, d-4] + g2*ljk[d-2, d-4],
                     g1*ljk[d-3, d-3] + g2*ljk[d-2, d-3]]
            row += 2 * [d-3]
            col += [d-3, d-2]
        
        elif i == d-2:
            data += [g1*ljk[d-2, d-4] + g2*ljk[d-1, d-4],
                     g1*ljk[d-2, d-3] + g2*ljk[d-1, d-3]]
            row += 2 * [d-2]
            col += [d-3, d-2]
                
        elif i == d-1: # g2=0
            data += [g1 * ljk[d-1, d-4], g1 * ljk[d-1, d-3]]
            row += 2 * [d-1]
            col += [d-3, d-2]

    return coo_matrix((data, (row, col)), shape = (d, d), dtype = 'float').tocsc()'''

"""
Steady state for 1D population
"""
cpdef np.ndarray steady_state_1D(int n, float N = 1.0, float gamma = 0.0, float h = 0.5, float theta = 1.0):
    cdef int d
    # dimensions of the sfs
    d = n+1
    
    # matrix for mutations
    u = np.float64(theta / 4.0)
    s = np.float64(gamma)

    B = np.zeros(d)
    B[1] = n * u

    # matrix for drift
    D = 1 / 4.0 / N * calcD(d)
    # jackknife matrices
    ljk = jk.calcJK13(int(d-1))
    ljk2 = jk.calcJK23(int(d-1))
    # matrix for selection
    S = s * h *calcS(d, ljk)
    
    S2 = s * (1-2.0*h) * calcS(d, ljk)
    # matrix for migration
    Mat = D+S+S2

    sfs = linalg.spsolve(Mat[1: d-1, 1: d-1], -B[1: d-1])
    sfs = np.insert(sfs, 0, 0.0)
    sfs = np.insert(sfs, d-1, 0.0)

    return sfs
