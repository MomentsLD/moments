import numpy as np

"""
Jackknife for selection in triallele model
"""

import numpy as np
import moments.Triallele.Numerics
from scipy.sparse import csr_matrix

"""
Jackknife extrapolation for Phi_{n+1} and Phi_{n+2} from Phi_n
Here, there are functions to find the six closest points to use for the extrapolation,
                          to compute the alpha terms for each extrap point, given n
                          and to compute the J matrices for each extrapolation
"""

### get closest six points for jackknife, (i/(n+1),j/(n+1))
def closest_ij_1(i, j, n):
    # sort by closest to farthest
    # I think we will need to have a spread of three grid points in each direction - a rectangular box leads to an A matrix with rank < 6
    fi, fj = i / (n + 1.0), j / (n + 1.0)
    possible_ij = []
    for ii in range(1, n):
        for jj in range(1, n - ii):
            possible_ij.append((ii, jj))
    possible_ij = np.array(possible_ij)
    smallests = np.argpartition(
        np.sum((np.array([fi, fj]) - possible_ij / (1.0 * n)) ** 2, axis=1), 6
    )[:6]
    smallest_set = np.array([possible_ij[k] for k in smallests])
    distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj]) ** 2, axis=1)
    order = distances.argsort()
    ordered_set = np.array([smallest_set[ii] for ii in order])
    # ensure that we have an index range of three in each direction
    # if we don't, drop the last (farthest) point, and get next closest until we have three points in each direction
    i_range, j_range = (
        np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
        np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
    )
    next_index = 7
    while i_range < 2 or j_range < 2:
        smallests = np.argpartition(
            np.sum((np.array([fi, fj]) - possible_ij / (1.0 * n)) ** 2, axis=1),
            next_index,
        )[:next_index]
        smallest_set = np.array([possible_ij[k] for k in smallests])
        distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj]) ** 2, axis=1)
        order = distances.argsort()
        new_ordered_set = np.array([smallest_set[ii] for ii in order])
        ordered_set[-1] = new_ordered_set[-1]
        i_range, j_range = (
            np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
            np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
        )
        next_index += 1
    return ordered_set


def compute_alphas_1(i, j, ordered_set, n):
    A = np.zeros((6, 6))
    b = np.zeros(6)
    A[0] = 1
    A[1] = ordered_set[:, 0] + 1.0
    A[2] = ordered_set[:, 1] + 1.0
    A[3] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
    A[4] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 1] + 1.0)
    A[5] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
    b[0] = (n + 1.0) / (n + 3)
    b[1] = (n + 1.0) / (n + 4.0) * (i + 1.0)
    b[2] = (n + 1.0) / (n + 4.0) * (j + 1.0)
    b[3] = (n + 1.0) / (n + 5.0) * (i + 1.0) * (i + 2.0)
    b[4] = (n + 1.0) / (n + 5.0) * (i + 1.0) * (j + 1.0)
    b[5] = (n + 1.0) / (n + 5.0) * (j + 1.0) * (j + 2.0)
    return np.dot(np.linalg.inv(A), b)


## now for the n+2 jackknife
def closest_ij_2(i, j, n):
    # sort by closest to farthest
    # I think we will need to have a spread of three grid points in each direction - a rectangular box leads to an A matrix with rank < 6
    fi, fj = i / (n + 2.0), j / (n + 2.0)
    possible_ij = []
    for ii in range(1, n):
        for jj in range(1, n - ii):
            possible_ij.append((ii, jj))
    possible_ij = np.array(possible_ij)
    smallests = np.argpartition(
        np.sum((np.array([fi, fj]) - possible_ij / (1.0 * n)) ** 2, axis=1), 6
    )[:6]
    smallest_set = np.array([possible_ij[k] for k in smallests])
    distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj]) ** 2, axis=1)
    order = distances.argsort()
    ordered_set = np.array([smallest_set[ii] for ii in order])
    # ensure that we have an index range of three in each direction
    # if we don't, drop the last (farthest) point, and get next closest until we have three points in each direction
    i_range, j_range = (
        np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
        np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
    )
    next_index = 7
    while i_range < 2 or j_range < 2:
        smallests = np.argpartition(
            np.sum((np.array([fi, fj]) - possible_ij / (1.0 * n)) ** 2, axis=1),
            next_index,
        )[:next_index]
        smallest_set = np.array([possible_ij[k] for k in smallests])
        distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj]) ** 2, axis=1)
        order = distances.argsort()
        new_ordered_set = np.array([smallest_set[ii] for ii in order])
        ordered_set[-1] = new_ordered_set[-1]
        i_range, j_range = (
            np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
            np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
        )
        next_index += 1
    return ordered_set


def compute_alphas_2(i, j, ordered_set, n):
    A = np.zeros((6, 6))
    b = np.zeros(6)
    A[0] = 1
    A[1] = ordered_set[:, 0] + 1.0
    A[2] = ordered_set[:, 1] + 1.0
    A[3] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
    A[4] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 1] + 1.0)
    A[5] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
    b[0] = (n + 1.0) * (n + 2.0) / ((n + 3.0) * (n + 4.0))
    b[1] = (n + 1.0) * (n + 2.0) / ((n + 4.0) * (n + 5.0)) * (i + 1.0)
    b[2] = (n + 1.0) * (n + 2.0) / ((n + 4.0) * (n + 5.0)) * (j + 1.0)
    b[3] = (n + 1.0) * (n + 2.0) / ((n + 5.0) * (n + 6.0)) * (i + 1.0) * (i + 2.0)
    b[4] = (n + 1.0) * (n + 2.0) / ((n + 5.0) * (n + 6.0)) * (i + 1.0) * (j + 1.0)
    b[5] = (n + 1.0) * (n + 2.0) / ((n + 5.0) * (n + 6.0)) * (j + 1.0) * (j + 2.0)
    return np.dot(np.linalg.inv(A), b)


def find_iprime_1D(n, i):
    # get iprime/n closest to i/(n+2)
    iis = np.arange(n + 1)
    ii = np.argmin(abs(iis / (1.0 * n) - i / (n + 2.0)))
    if ii < 2:
        ii = 2
    if ii > n - 2:
        ii = n - 2
    return ii


def get_alphas_1D(ii, i, n):
    A = np.zeros((3, 3))
    A[0] = 1
    A[1] = ii + np.arange(3)
    A[2] = (ii + np.arange(3)) * (ii + np.arange(1, 4))
    b = np.array(
        [
            (n + 1.0) / (n + 3),
            (n + 1.0) * (n + 2) * (i + 1) / ((n + 3) * (n + 4)),
            (n + 1.0) * (n + 2) * (i + 1) * (i + 2) / ((n + 4) * (n + 5)),
        ]
    )
    return np.dot(np.linalg.inv(A), b)


# compute the quadratic two-dim Jackknife extrapolation for Phi_n to Phi_{n+2}
# i,j are the indices in the n+1 spectrum (just for interior points)
def calcJK_2(n):
    # size of J is size of n+1 spectrum x size of n spectrum
    # J = np.zeros(((n+3)*(n+4)/2,(n+1)*(n+2)/2))
    row = []
    col = []
    data = []

    for i in range(1, n + 2):
        for j in range(1, n + 2 - i):
            ordered_set = closest_ij_2(i, j, n)
            alphas = compute_alphas_2(i, j, ordered_set, n)
            index2 = moments.Triallele.Numerics.get_index(n + 2, i, j)
            for pair, alpha in zip(ordered_set, alphas):
                index = moments.Triallele.Numerics.get_index(n, pair[0], pair[1])
                # J[index2,index] = alpha
                row.append(index2)
                col.append(index)
                data.append(alpha)

    # jackknife for the biallelic edges (i=0, j=1:n, and j=0, i=1:n)
    # first for j = 0
    j = 0
    for i in range(1, n + 2):
        this_ind = moments.Triallele.Numerics.get_index(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        # J[this_ind, moments.Triallele.Numerics.get_index(n,ii-1,j)] = alphas[0]
        # J[this_ind, moments.Triallele.Numerics.get_index(n,ii,j)] = alphas[1]
        # J[this_ind, moments.Triallele.Numerics.get_index(n,ii+1,j)] = alphas[2]
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, ii - 1, 0))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, ii, 0))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, ii + 1, 0))
        data.append(alphas[2])

    i = 0
    for j in range(1, n + 2):
        this_ind = moments.Triallele.Numerics.get_index(n + 2, i, j)
        jj = find_iprime_1D(n, j)
        alphas = get_alphas_1D(jj, j, n)
        # J[this_ind, moments.Triallele.Numerics.get_index(n,i,jj-1)] = alphas[0]
        # J[this_ind, moments.Triallele.Numerics.get_index(n,i,jj)] = alphas[1]
        # J[this_ind, moments.Triallele.Numerics.get_index(n,i,jj+1)] = alphas[2]
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, 0, jj - 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, 0, jj))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, 0, jj + 1))
        data.append(alphas[2])

    # jackknife along diagonal - 1D jk
    for i in range(1, n + 2):
        j = n + 2 - i
        this_ind = moments.Triallele.Numerics.get_index(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, ii - 1, n - ii + 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, ii, n - ii))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(moments.Triallele.Numerics.get_index(n, ii + 1, n - ii - 1))
        data.append(alphas[2])

    return csr_matrix(
        (data, (row, col)),
        shape=(int((n + 3) * (n + 4) / 2), int((n + 1) * (n + 2) / 2)),
    )
