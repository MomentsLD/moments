import numpy as np
cimport numpy as np
from scipy.sparse import coo_matrix

import Jackknife as jk

"""
Functions to build the matrices we need for the linear system
The code below is written for 2D cases as we split by plans for the integration.
For each component (drift, selection, migration) we consider separately the 2 dimensions.
"""

"""
Matrices for forward and backward mutations
dims = numpy.array([n1+1,n2+1])
"""
# mutations in the first population:
cpdef calcB_FB1(np.ndarray dims, np.float64_t u, np.float64_t v):
    cdef int d, d1, d2, i, index
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d1, d2 = dims
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        # index in the first dimension
        index = i // d2
        if index > 0:
            data += [u * (d1-index), -v * index]
            row += 2 * [i]
            col += [i - d2, i]
        if index < d1 - 1:
            data += [-u * (d1-index-1), v * (index+1)]
            row += 2 * [i]
            col += [i, i + d2]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

# mutations in the second population:
cpdef calcB_FB2(np.ndarray dims, np.float64_t u, np.float64_t v):
    cdef int d, d2, i, index
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d2 = dims[1]
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        # index in the second dimension
        index = i % d2
        if index > 0:
            data += [u * (d2-index), -v * index]
            row += 2 * [i]
            col += [i - 1, i]
        if index < d2 - 1:
            data += [-u * (d2-index-1), v * (index+1)]
            row += 2 * [i]
            col += [i, i + 1]
    
    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

"""
Matrices for drift
dims = numpy.array([n1+1,n2+1])
"""
# drift along the first axis :
cpdef calcD1(np.ndarray dims):
    cdef int d, d1, d2, i, index
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d1, d2 = dims
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        # index in the first dimension
        index = i // d2
        if index > 1:
            data.append((index-1) * (d1-index))
            row.append(i)
            col.append(i - d2)
        if index < d1 - 2:
            data.append((index+1) * (d1-index-2))
            row.append(i)
            col.append(i + d2)
        if index > 0 and index < d1 - 1:
            data.append(-2 * index * (d1-index-1))
            row.append(i)
            col.append(i)
    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

# drift along the second axis :
cpdef calcD2(np.ndarray dims):
    cdef int d, d2, i, index
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d2 = dims[1]
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    # loop over the fs elements:
    for i in range(d):
        # index in the second dimension
        index = i % d2
        if index > 1:
            data.append((index-1) * (d2-index))
            row.append(i)
            col.append(i - 1)
        if index < d2 - 2:
            data.append((index+1) * (d2-index-2))
            row.append(i)
            col.append(i + 1)
        if index > 0 and index < d2 - 1:
            data.append(-2 * index * (d2-index-1))
            row.append(i)
            col.append(i)
    
    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

"""
Matrices for selection with order 3 JK
dims = numpy.array([n1+1,n2+1])
ljk is the Jacknife array corresponding to the concerned population size:
    ljk=ljk(pop1) in calcS_1, calcS2_1 and ljk=ljk(pop2) in calcS_2, calcS2_2
s and h are the coefficients for selection and dominance in the concerned population.
"""
# selection along the first dimension with h1 = 0.5
cpdef calcS_1(np.ndarray dims, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int d, d1, d2, i, j, k, i_bis, i_ter
    cdef np.float64_t g1, g2
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d1, d2 = dims
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    
    for k in range(d):
        # 2D index of the current variable
        i, j = k // d2, k % d2
        i_bis = jk.index_bis(i, d1 - 1)
        i_ter = jk.index_bis(i + 1, d1 - 1)
        # coefficients
        g1 = i * (d1-i) / np.float64(d1)
        g2 = -(i+1) * (d1-1-i) / np.float64(d1)
        if i < d1 - 1:
            data += [g1 * ljk[i - 1, i_bis - 1], g1 * ljk[i - 1, i_bis - 2],
                    g1 * ljk[i - 1, i_bis], g2 * ljk[i, i_ter - 1],
                    g2 * ljk[i, i_ter - 2], g2 * ljk[i, i_ter]]
            row += 6 * [k]
            col += [i_bis*d2 + j, (i_bis-1)*d2 + j, (i_bis+1)*d2 + j,
                    i_ter*d2 + j, (i_ter-1)*d2 + j, (i_ter+1)*d2 + j]
            
        if i == d1 - 1: # g2=0
            data += [g1 * ljk[i - 1, i_bis - 1], g1 * ljk[i - 1, i_bis - 2],
                     g1 * ljk[i - 1, i_bis]]
            row += 3 * [k]
            col += [i_bis*d2 + j, (i_bis-1)*d2 + j, (i_bis+1)*d2 + j]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

# selection along the second dimension with h2 = 0.5
cpdef calcS_2(np.ndarray dims, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int d, d2, i, j, k, j_bis, j_ter
    cdef np.float64_t g1, g2
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d2 = dims[1]
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for k in range(d):
        # 2D index of the current variable
        i, j = k // d2, k % d2
        j_bis = jk.index_bis(j, d2 - 1)
        j_ter = jk.index_bis(j + 1, d2 - 1)
        # coefficients
        g1 = j * (d2-j) / np.float64(d2)
        g2 = -(j+1) * (d2-1-j) / np.float64(d2)
        if j < d2 - 1:
            data += [g1 * ljk[j - 1, j_bis - 1], g1 * ljk[j - 1, j_bis - 2],
                    g1 * ljk[j - 1, j_bis], g2 * ljk[j, j_ter - 1],
                    g2 * ljk[j, j_ter - 2], g2 * ljk[j, j_ter]]
            row += 6 * [k]
            col += [i*d2 + j_bis, i*d2 + j_bis - 1, i*d2 + j_bis + 1,
                    i*d2 + j_ter, i*d2 + j_ter - 1, i*d2 + j_ter + 1]
            
        if j == d2 - 1: # g2=0
            data += [g1 * ljk[j - 1, j_bis - 1], g1 * ljk[j - 1, j_bis - 2],
                     g1 * ljk[j - 1, j_bis]]
            row += 3 * [k]
            col += [i*d2 + j_bis, i*d2 + j_bis - 1, i*d2 + j_bis + 1]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

# selection along the first dimension, part related to h1 != 0.5
# ljk is a 2-jumps jackknife
cpdef calcS2_1(np.ndarray dims, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int d, d1, d2, k, i, j, i_ter, i_qua
    cdef np.float64_t g1, g2
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d1, d2 = dims
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for k in range(d):
        # 2D index of the current variable
        i, j = k // d2, k % d2
        i_ter = jk.index_bis(i + 1, d1 - 1)
        i_qua = jk.index_bis(i + 2, d1 - 1)
        g1 = (i+1) / np.float64(d1) / (d1+1) * i * (d1-i)
        g2 = -(i+1) / np.float64(d1) / (d1+1) * (i+2) * (d1-1-i)

        if i < d1 - 1:
            data += [g1 * ljk[i, i_ter - 1], g1 * ljk[i, i_ter - 2],
                    g1 * ljk[i, i_ter], g2 * ljk[i + 1, i_qua - 1],
                    g2 * ljk[i + 1, i_qua - 2], g2 * ljk[i + 1, i_qua]]
            row += 6 * [k]
            col += [i_ter*d2 + j, (i_ter-1)*d2 + j, (i_ter+1)*d2 + j,
                    i_qua*d2 + j, (i_qua-1)*d2 + j, (i_qua+1)*d2 + j]
            
        if i == d1 - 1: # g2=0
            data += [g1 * ljk[i, i_ter - 1], g1 * ljk[i, i_ter - 2],
                     g1 * ljk[i, i_ter]]
            row += 3 * [k]
            col += [i_ter*d2 + j, (i_ter-1)*d2 + j, (i_ter+1)*d2 + j]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

# selection along the second dimension, part related to h2 != 0.5
# ljk is a 2-jumps jackknife
cpdef calcS2_2(np.ndarray dims, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int d, d1, d2, k, i, j, j_ter, j_qua
    cdef np.float64_t g1, g2
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d2 = dims[1]
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for k in range(d):
        # 2D index of the current variable
        i, j = k // d2, k % d2
        j_ter = jk.index_bis(j + 1, d2 - 1)
        j_qua = jk.index_bis(j + 2, d2 - 1)
        g1 = (j+1) / np.float64(d2) / (d2+1) * j * (d2-j)
        g2 = -(j+1) / np.float64(d2) / (d2+1) * (j+2) * (d2-1-j)

        if j < d2 - 1:
            data += [g1 * ljk[j, j_ter - 1], g1 * ljk[j, j_ter - 2],
                    g1 * ljk[j, j_ter], g2 * ljk[j + 1, j_qua - 1],
                    g2 * ljk[j + 1, j_qua - 2], g2 * ljk[j + 1, j_qua]]
            row += 6 * [k]
            col += [i*d2 + j_ter, i*d2 + j_ter - 1, i*d2 + j_ter + 1,
                    i*d2 + j_qua, i*d2 + j_qua - 1, i*d2 + j_qua + 1]
            
        if j == d2 - 1: # g2=0
            data += [g1 * ljk[j, j_ter - 1], g1 * ljk[j, j_ter - 2],
                     g1 * ljk[j, j_ter]]
            row += 3 * [k]
            col += [i*d2 + j_ter, i*d2 + j_ter - 1, i*d2 + j_ter + 1]

    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()


# over and under dominance
cpdef calcUnderdominance_1(np.ndarray dims, np.ndarray[np.float64_t, ndim=2] ljk):
    cdef int d, d2, k, i, j, i0, i1, i2
    cdef np.float64_t g1, g2
    cdef list data, row, col

    d = int(np.prod(dims))
    d1, d2 = dims
    
    data = []
    row = []
    col = []

    for k in range(d):
        i, j = k // d2, k % d2
        i0 = jk.index_bis(i, d1 - 1)
        i1 = jk.index_bis(i + 1, d1 - 1)
        i2 = jk.index_bis(i + 2, d1 - 1)

        # first term
        fac = 1. / (d1 + 1.) / np.float64(d1)
        g1 = fac * i * (d1 - i + 1) * (d1 - i)
        g2 = -fac * (i + 1) * (d1 - i) * (d1 - i - 1)
        if i > 0 and i < d1 - 1:
            data += [
                g1 * ljk[i - 1, i0 - 1], g1 * ljk[i - 1, i0 - 2], g1 * ljk[i - 1, i0],
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 6 * [k]
            col += [i0 * d2 + j, (i0 - 1) * d2 + j, (i0 + 1) * d2 + j,
                    i1 * d2 + j, (i1 - 1) * d2 + j, (i1 + 1) * d2 + j]
        elif i == 0:
            data += [
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 3 * [k]
            col += [i1 * d2 + j, (i1 - 1) * d2 + j, (i1 + 1) * d2 + j]
        elif i == d1 - 1:
            data += [
                g1 * ljk[i - 1, i0 - 1], g1 * ljk[i - 1, i0 - 2], g1 * ljk[i - 1, i0]
            ]
            row += 3 * [k]
            col += [i0 * d2 + j, (i0 - 1) * d2 + j, (i0 + 1) * d2 + j]
        
        # second term
        g1 = fac * (i + 2) * (i + 1) * (d1 - 1 - i)
        g2 = -fac * (i + 1) * i * (d1 - i)
        if i > 0 and i < d1 - 1:
            data += [
                g1 * ljk[i + 1, i2 - 1], g1 * ljk[i + 1, i2 - 2], g1 * ljk[i + 1, i2],
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 6 * [k]
            col += [i2 * d2 + j, (i2 - 1) * d2 + j, (i2 + 1) * d2 + j,
                    i1 * d2 + j, (i1 - 1) * d2 + j, (i1 + 1) * d2 + j]
        elif i == d1 - 1:
            data += [
                g2 * ljk[i, i1 - 1], g2 * ljk[i, i1 - 2], g2 * ljk[i, i1]
            ]
            row += 3 * [k]
            col += [i1 * d2 + j, (i1 - 1) * d2 + j, (i1 + 1) * d2 + j]
        elif i == 0:
            data += [
                g1 * ljk[i + 1, i2 - 1], g1 * ljk[i + 1, i2 - 2], g1 * ljk[i + 1, i2]
            ]
            row += 3 * [k]
            col += [i2 * d2 + j, (i2 - 1) * d2 + j, (i2 + 1) * d2 + j]

    M = coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()
    return M


cpdef calcUnderdominance_2(np.ndarray dims, np.ndarray[np.float64_t, ndim=2] ljk):
    cdef int d, d2, k, i, j, i0, i1, i2
    cdef np.float64_t g1, g2
    cdef list data, row, col

    d = int(np.prod(dims))
    d2 = dims[1]
    
    data = []
    row = []
    col = []

    for k in range(d):
        i, j = k // d2, k % d2
        j0 = jk.index_bis(j, d2 - 1)
        j1 = jk.index_bis(j + 1, d2 - 1)
        j2 = jk.index_bis(j + 2, d2 - 1)

        # first term
        fac = 1. / (d2 + 1.) / np.float64(d2)
        g1 = fac * j * (d2 - j + 1) * (d2 - j)
        g2 = -fac * (j + 1) * (d2 - j) * (d2 - j - 1)
        if j > 0 and j < d2 - 1:
            data += [
                g1 * ljk[j - 1, j0 - 1], g1 * ljk[j - 1, j0 - 2], g1 * ljk[j - 1, j0],
                g2 * ljk[j, j1 - 1], g2 * ljk[j, j1 - 2], g2 * ljk[j, j1]
            ]
            row += 6 * [k]
            col += [i * d2 + j0, i * d2 + (j0 - 1), i * d2 + (j0 + 1),
                    i * d2 + j1, i * d2 + (j1 - 1), i * d2 + (j1 + 1)]
        elif j == 0:
            data += [
                g2 * ljk[j, j1 - 1], g2 * ljk[j, j1 - 2], g2 * ljk[j, j1]
            ]
            row += 3 * [k]
            col += [i * d2 + j1, i * d2 + (j1 - 1), i * d2 + (j1 + 1)]
        elif j == d2 - 1:
            data += [
                g1 * ljk[j - 1, j0 - 1], g1 * ljk[j - 1, j0 - 2], g1 * ljk[j - 1, j0]
            ]
            row += 3 * [k]
            col += [i * d2 + j0, i * d2 + (j0 - 1), i * d2 + (j0 + 1)]
        
        # second term
        g1 = fac * (j + 2) * (j + 1) * (d2 - 1 - j)
        g2 = -fac * (j + 1) * j * (d2 - j)
        if j > 0 and j < d2 - 1:
            data += [
                g1 * ljk[j + 1, j2 - 1], g1 * ljk[j + 1, j2 - 2], g1 * ljk[j + 1, j2],
                g2 * ljk[j, j1 - 1], g2 * ljk[j, j1 - 2], g2 * ljk[j, j1]
            ]
            row += 6 * [k]
            col += [i * d2 + j2, i * d2 + (j2 - 1), i * d2 + (j2 + 1),
                    i * d2 + j1, i * d2 + (j1 - 1), i * d2 + (j1 + 1)]
        elif j == d2 - 1:
            data += [
                g2 * ljk[j, j1 - 1], g2 * ljk[j, j1 - 2], g2 * ljk[j, j1]
            ]
            row += 3 * [k]
            col += [i * d2 + j1, i * d2 + (j1 - 1), i * d2 + (j1 + 1)]
        elif j == 0:
            data += [
                g1 * ljk[j + 1, j2 - 1], g1 * ljk[j + 1, j2 - 2], g1 * ljk[j + 1, j2]
            ]
            row += 3 * [k]
            col += [i * d2 + j2, i * d2 + (j2 - 1), i * d2 + (j2 + 1)]

    M = coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()
    return M


"""
Matrices for migration with order 3 JK
dims = numpy.array([n1+1,n2+1])
ljk is the Jacknife array corresponding to the concerned population size: 
    ljk=ljk(pop2) in calcM1 and ljk=ljk(pop1) in calcM2
m is the migration rate: m=m12 in calcM1 and m=m21 in calcM2
"""
cpdef calcM_1(np.ndarray dims, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int d, d1, d2, i, j, k, i_ter
    cdef np.float64_t c, coeff1, coef2, coeff3
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d1, d2 = dims
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for k in range(d):
        # 2D index of the current variable
        i, j = k // d2, k % d2
        j_ter = jk.index_bis(j + 1, d2 - 1)

        c = (j+1) / np.float64(d2)
        coeff1 = (2*i-(d1-1)) * c
        coeff2 = (d1-i) * c
        coeff3 = -(i+1) * c
                
        data.append(-i)
        row.append(k)
        col.append(k)

        if i < d1 - 1:
            data.append(i + 1)
            row.append(k)
            col.append(k + d2)
                
        if j < d2 - 1:
            data += [coeff1 * ljk[j, j_ter - 2], coeff1 * ljk[j, j_ter - 1],
                     coeff1 * ljk[j, j_ter]]
            row += 3 * [k]
            col += [i*d2 + j_ter - 1, i*d2 + j_ter, i*d2 + j_ter + 1]
            if i > 0:
                data +=[coeff2 * ljk[j, j_ter - 2], coeff2 * ljk[j, j_ter - 1],
                        coeff2 * ljk[j, j_ter]]
                row += 3 * [k]
                col += [(i-1)*d2 + j_ter - 1, (i-1)*d2 + j_ter, (i-1)*d2 + j_ter + 1]
            if i < d1 - 1:
                data += [coeff3 * ljk[j, j_ter - 2], coeff3 * ljk[j, j_ter - 1],
                         coeff3 * ljk[j, j_ter]]
                row += 3 * [k]
                col += [(i+1)*d2 + j_ter - 1, (i+1)*d2 + j_ter, (i+1)*d2 + j_ter + 1]
            
        elif j == d2 - 1:
            data += [coeff1, -coeff1 / d2 * ljk[j - 1, j_ter - 2],
                    -coeff1 / d2 * ljk[j - 1, j_ter - 1],
                    -coeff1 / d2 * ljk[j - 1, j_ter]]
            row += 4 * [k]
            col += [k, i*d2 + j_ter - 1, i*d2 + j_ter, i*d2 + j_ter + 1]
                             
            if i > 0:
                data += [coeff2, -coeff2 / d2 * ljk[j - 1, j_ter - 2],
                        -coeff2 / d2 * ljk[j - 1, j_ter - 1],
                        -coeff2 / d2 * ljk[j - 1, j_ter]]
                row += 4 * [k]
                col += [k - d2, (i-1)*d2 + j_ter - 1,
                        (i-1)*d2 + j_ter, (i-1)*d2 + j_ter + 1]
                                     
            if i < d1 - 1:
                data += [coeff3, -coeff3 / d2 * ljk[j - 1, j_ter - 2],
                        -coeff3 / d2 * ljk[j - 1, j_ter - 1],
                        -coeff3 / d2 * ljk[j - 1, j_ter]]
                row += 4 * [k]
                col += [k + d2, (i+1)*d2 + j_ter - 1,
                        (i+1)*d2 + j_ter, (i+1)*d2 + j_ter + 1]
    
    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()

cpdef calcM_2(np.ndarray dims, np.ndarray[np.float64_t, ndim = 2] ljk):
    cdef int d, d1, d2, i, j, k, i_ter
    cdef np.float64_t c, coeff1, coef2, coeff3
    cdef list data, row, col
    # number of degrees of freedom
    d = int(np.prod(dims))
    d1, d2 = dims
 
    # arrays for the creation of the sparse (coo) matrix
    data = []
    row = []
    col = []
    for k in range(d):
        # 2D index of the current variable
        i, j = k // d2, k % d2
        i_ter = jk.index_bis(i + 1, d1 - 1)
        c = (i+1) / np.float64(d1)
        coeff1 = (2*j-(d2-1)) * c
        coeff2 = (d2-j) * c
        coeff3 = -(j+1) * c
      
        data.append(-j)
        row.append(k)
        col.append(k)
                
        if j < d2 - 1:
            data.append(j + 1)
            row.append(k)
            col.append(k + 1)
                
        if i < d1 - 1:
            data += [coeff1 * ljk[i, i_ter - 2], coeff1 * ljk[i, i_ter - 1],
                     coeff1 * ljk[i, i_ter]]
            row += 3 * [k]
            col += [(i_ter-1)*d2 + j, i_ter*d2 + j, (i_ter+1)*d2 + j]
            if j > 0:
                data +=[coeff2 * ljk[i, i_ter - 2], coeff2 * ljk[i, i_ter - 1],
                        coeff2 * ljk[i, i_ter]]
                row += 3 * [k]
                col += [(i_ter-1)*d2 + j - 1, i_ter*d2 + j - 1, (i_ter+1)*d2 + j - 1]
            if j < d2 - 1:
                data += [coeff3 * ljk[i, i_ter - 2], coeff3 * ljk[i, i_ter - 1],
                         coeff3 * ljk[i, i_ter]]
                row += 3 * [k]
                col += [(i_ter-1)*d2 + j + 1, i_ter*d2 + j + 1, (i_ter+1)*d2 + j + 1]
            
        elif i == d1 - 1:
            data += [coeff1, -coeff1 / d1 * ljk[i - 1, i_ter - 2],
                    -coeff1 / d1 * ljk[i - 1, i_ter - 1],
                    -coeff1 / d1 * ljk[i - 1, i_ter]]
            row += 4 * [k]
            col += [k, (i_ter-1)*d2 + j, i_ter*d2 + j, (i_ter+1)*d2 + j]
                             
            if j > 0:
                data += [coeff2, -coeff2 / d1 * ljk[i - 1, i_ter - 2],
                        -coeff2 / d1 * ljk[i - 1, i_ter - 1],
                        -coeff2 / d1 * ljk[i - 1, i_ter]]
                row += 4 * [k]
                col += [k - 1, (i_ter-1)*d2 + j - 1,
                        i_ter*d2 + j - 1, (i_ter+1)*d2 + j - 1]
                                     
            if j < d2 - 1:
                data += [coeff3, -coeff3 / d1 * ljk[i - 1, i_ter - 2],
                        -coeff3 / d1 * ljk[i - 1, i_ter - 1],
                        -coeff3 / d1 * ljk[i - 1, i_ter]]
                row += 4 * [k]
                col += [k + 1, (i_ter-1)*d2 + j + 1,
                        i_ter*d2 + j + 1, (i_ter+1)*d2 + j + 1]
    
    return coo_matrix((data, (row, col)), shape=(d, d), dtype='float').tocsc()
