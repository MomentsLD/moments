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
            col.append(i + d2)
            row.append(i)
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
            col.append(i + 1)
            row.append(i)
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
    cdef int d, d2, k, i, j, j_ter, j_qua
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
