import numpy as np
from scipy.sparse import coo_matrix

"""
Mutation matrices for reversible mutations, given spectrum dimension, u and v
"""

# three populations
def calc_FB_3pop(dims, u, v):
    d = int(np.prod(dims))
    d1, d2, d3 = dims
    # arrays for the creation of the sparse (coo) matrices
    data1 = []
    row1 = []
    col1 = []
    data2 = []
    row2 = []
    col2 = []
    data3 = []
    row3 = []
    col3 = []
    for i in range(d):
        # index in each dimension
        index1 = i // (d2*d3)
        index2 = i % (d2*d3) // d3
        index3 = i % (d2*d3) % d3
        if index1 > 0:
            data1 += [u[0] * (d1-index1), -v[0] * index1]
            row1 += 2 * [i]
            col1 += [i - d2*d3, i]
        if index1 < d1-1:
            data1 += [-u[0] * (d1-index1-1), v[0] * (index1+1)]
            row1 += 2 * [i]
            col1 += [i, i + d2*d3]
        if index2 > 0:
            data2 += [u[1] * (d2-index2), -v[1] * index2]
            row2 += 2 * [i]
            col2 += [i - d3, i]
        if index2 < d2-1:
            data2 += [-u[1] * (d2-index2-1), v[1] * (index2+1)]
            row2 += 2 * [i]
            col2 += [i, i+d3]
        if index3 > 0:
            data3 += [u[2] * (d3-index3), -v[2] * index3]
            row3 += 2 * [i]
            col3 += [i-1, i]
        if index3 < d3-1:
            data3 += [-u[2] * (d3-index3-1), v[2] * (index3+1)]
            row3 += 2 * [i]
            col3 += [i, i+1]
    return [coo_matrix((data1, (row1, col1)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data2, (row2, col2)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data3, (row3, col3)), shape=(d, d), dtype='float').tocsc()]


# four populations
def calc_FB_4pop(dims, u, v):
    d = int(np.prod(dims))
    d1, d2, d3, d4 = dims
    # arrays for the creation of the sparse (coo) matrices
    data1 = []
    row1 = []
    col1 = []
    data2 = []
    row2 = []
    col2 = []
    data3 = []
    row3 = []
    col3 = []
    data4 = []
    row4 = []
    col4 = []
    for i in range(d):
        # index in each dimension
        index1 = i // (d2*d3*d4)
        index2 = i % (d2*d3*d4) // (d3*d4)
        index3 = i % (d2*d3*d4) % (d3*d4) // d4
        index4 = i % (d2*d3*d4) % (d3*d4) % d4
        if index1 > 0:
            data1 += [u[0] * (d1-index1), -v[0] * index1]
            row1 += 2 * [i]
            col1 += [i - d2*d3*d4, i]
        if index1 < d1-1:
            data1 += [-u[0] * (d1-index1-1), v[0] * (index1+1)]
            row1 += 2 * [i]
            col1 += [i, i + d2*d3*d4]
        
        if index2 > 0:
            data2 += [u[1] * (d2-index2), -v[1] * index2]
            row2 += 2 * [i]
            col2 += [i - d3*d4, i]
        if index2 < d2-1:
            data2 += [-u[1] * (d2-index2-1), v[1] * (index2+1)]
            row2 += 2 * [i]
            col2 += [i, i + d3*d4]
        
        if index3 > 0:
            data3 += [u[2] * (d3-index3), -v[2] * index3]
            row3 += 2 * [i]
            col3 += [i - d4, i]
        if index3 < d3-1:
            data3 += [-u[2] * (d3-index3-1), v[2] * (index3+1)]
            row3 += 2 * [i]
            col3 += [i, i + d4]
        
        if index4 > 0:
            data4 += [u[3] * (d4-index4), -v[3] * index4]
            row4 += 2 * [i]
            col4 += [i - 1, i]
        if index4 < d4-1:
            data4 += [-u[3] * (d4-index4-1), v[3] * (index4+1)]
            row4 += 2 * [i]
            col4 += [i, i + 1]
    
    return [coo_matrix((data1, (row1, col1)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data2, (row2, col2)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data3, (row3, col3)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data4, (row4, col4)), shape=(d, d), dtype='float').tocsc()]

# 5 pops
def calc_FB_5pop(dims, u, v):
    d = int(np.prod(dims))
    d1, d2, d3, d4, d5 = dims
    # arrays for the creation of the sparse (coo) matrices
    data1 = []
    row1 = []
    col1 = []
    data2 = []
    row2 = []
    col2 = []
    data3 = []
    row3 = []
    col3 = []
    data4 = []
    row4 = []
    col4 = []
    data5 = []
    row5 = []
    col5 = []
    for i in range(d):
        # index in each dimension
        index1 = i // (d2*d3*d4*d5)
        index2 = i % (d2*d3*d4*d5) // (d3*d4*d5)
        index3 = i % (d2*d3*d4*d5) % (d3*d4*d5) // (d4*d5)
        index4 = i % (d2*d3*d4*d5) % (d3*d4*d5) % (d4*d5) // d5
        index5 = i % (d2*d3*d4*d5) % (d3*d4*d5) % (d4*d5) % d5
        if index1 > 0:
            data1 += [u[0] * (d1-index1), -v[0] * index1]
            row1 += 2 * [i]
            col1 += [i - d2*d3*d4*d5, i]
        if index1 < d1-1:
            data1 += [-u[0] * (d1-index1-1), v[0] * (index1+1)]
            row1 += 2 * [i]
            col1 += [i, i + d2*d3*d4*d5]
        
        if index2 > 0:
            data2 += [u[1] * (d2-index2), -v[1] * index2]
            row2 += 2 * [i]
            col2 += [i - d3*d4*d5, i]
        if index2 < d2-1:
            data2 += [-u[1] * (d2-index2-1), v[1] * (index2+1)]
            row2 += 2 * [i]
            col2 += [i, i + d3*d4*d5]
        
        if index3 > 0:
            data3 += [u[2] * (d3-index3), -v[2] * index3]
            row3 += 2 * [i]
            col3 += [i - d4*d5, i]
        if index3 < d3-1:
            data3 += [-u[2] * (d3-index3-1), v[2] * (index3+1)]
            row3 += 2 * [i]
            col3 += [i, i + d4*d5]
        
        if index4 > 0:
            data4 += [u[3] * (d4-index4), -v[3] * index4]
            row4 += 2 * [i]
            col4 += [i - d5, i]
        if index4 < d4-1:
            data4 += [-u[3] * (d4-index4-1), v[3] * (index4+1)]
            row4 += 2 * [i]
            col4 += [i, i + d5]
    
        if index5 > 0:
            data5 += [u[4] * (d5-index5), -v[4] * index5]
            row5 += 2 * [i]
            col5 += [i - 1, i]
        if index5 < d5-1:
            data5 += [-u[4] * (d5-index5-1), v[4] * (index5+1)]
            row5 += 2 * [i]
            col5 += [i, i + 1]

    return [coo_matrix((data1, (row1, col1)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data2, (row2, col2)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data3, (row3, col3)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data4, (row4, col4)), shape=(d, d), dtype='float').tocsc(),
            coo_matrix((data5, (row5, col5)), shape=(d, d), dtype='float').tocsc()]

