import numpy as np
import math
import scipy.misc as misc

#-----------------------------------------
# Usefull functions for afs manipulations:
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