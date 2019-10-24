STUFF = "Hi"

import numpy as np
cimport numpy as np

"""
Functions to tally up genotype counts
"""


cpdef tally_sparse(dict G1, dict G2, int n, missing=False):
    """
    G1 and G2 are dictionaries with sample indices of genotypes 1 and 2
    and -1 if missing is True
    n is the diploid sample size
    """
    cdef int n22, n21, n20, n2m, n12, n11, n10, n1m, n02, n01, n00, nm1, nm2, nm
    
    if missing == True:
        # account for missing genotypes
        n22 = (G1[2] & G2[2]).__len__()
        n21 = (G1[2] & G2[1]).__len__()
        n2m = (G1[2] & G2[-1]).__len__()
        n20 = (G1[2]).__len__()-n22-n21-n2m
        n12 = (G1[1] & G2[2]).__len__()
        n11 = (G1[1] & G2[1]).__len__()
        n1m = (G1[1] & G2[-1]).__len__()
        n10 = (G1[1]).__len__()-n12-n11-n1m
        nm2 = (G1[-1] & G2[2]).__len__()
        nm1 = (G1[-1] & G2[1]).__len__()
        n02 = (G2[2]).__len__()-n22-n12-nm2
        n01 = (G2[1]).__len__()-n21-n11-nm1
        # total possible is n-len(set of either missing)
        nm = len(G1[-1].union(G2[-1]))
        n00 = (n-nm)-n22-n21-n20-n12-n11-n10-n02-n01
    else:
        n22 = (G1[2] & G2[2]).__len__()
        n21 = (G1[2] & G2[1]).__len__()
        n20 = (G1[2]).__len__()-n22-n21
        n12 = (G1[1] & G2[2]).__len__()
        n11 = (G1[1] & G2[1]).__len__()
        n10 = (G1[1]).__len__()-n12-n11
        n02 = (G2[2]).__len__()-n22-n12
        n01 = (G2[1]).__len__()-n21-n11
        n00 = n-n22-n21-n20-n12-n11-n10-n02-n01
    return (n22, n21, n20, n12, n11, n10, n02, n01, n00)


cpdef count_genotypes_sparse(dict G_dict, int n, missing=False):
    """
    Similar to count_genotypes, but using the sparse genotype representation instead
    """
    cdef int L = len(G_dict)
    
    cdef np.ndarray[np.int32_t, ndim=2] Counts = np.empty((L*(L-1)//2, 9), dtype=np.int32)
    cdef int c = 0
    cdef int i,j
    
    for i in range(L-1):
        for j in range(i+1,L):
            Counts[c] = tally_sparse(G_dict[i], G_dict[j], n, missing=missing)
            c += 1
    return Counts

cpdef count_genotypes_between_sparse(dict G_dict1, dict G_dict2, int n, missing=False):
    cdef int L1 = len(G_dict1)
    cdef int L2 = len(G_dict2)
    
    cdef np.ndarray[np.int32_t, ndim=2] Counts = np.empty((L1*L2, 9), dtype=np.int32)
    cdef int c = 0
    cdef int i,j
    
    for i in range(L1):
        for j in range(L2):
            Counts[c] = tally_sparse(G_dict1[i], G_dict2[j], n, missing=missing)
            c += 1
    
    return Counts



