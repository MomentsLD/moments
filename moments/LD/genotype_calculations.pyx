STUFF = "Hi" # otherwise cython craps out on me...

import numpy as np
cimport numpy as np

"""
Functions to take genotype counts and compute various two-locus statistics
"""

"""
Single population statistics
"""

cpdef compute_D(np.ndarray[np.int32_t, ndim=2] Counts):
    """
    D for block of genotypes counted from count_genotypes(Gs)
    """
    cdef np.ndarray[np.int32_t, ndim=1] n1 = Counts[:,0]
    cdef np.ndarray[np.int32_t, ndim=1] n2 = Counts[:,1]
    cdef np.ndarray[np.int32_t, ndim=1] n3 = Counts[:,2]
    cdef np.ndarray[np.int32_t, ndim=1] n4 = Counts[:,3]
    cdef np.ndarray[np.int32_t, ndim=1] n5 = Counts[:,4]
    cdef np.ndarray[np.int32_t, ndim=1] n6 = Counts[:,5]
    cdef np.ndarray[np.int32_t, ndim=1] n7 = Counts[:,6]
    cdef np.ndarray[np.int32_t, ndim=1] n8 = Counts[:,7]
    cdef np.ndarray[np.int32_t, ndim=1] n9 = Counts[:,8]
    cdef np.ndarray[np.int64_t, ndim=1] n
    n = np.sum(Counts, axis=1)
    cdef np.ndarray[np.float_t, ndim=1] numer
    cdef np.ndarray[np.float_t, ndim=1] denom

    cdef int i

    denom = 0.*n
    numer = 0.*n
    for i in range(len(n)):
        numer[i] = (-(n2[i]*n4[i])/4. - (n3[i]*n4[i])/2. + (n1[i]*n5[i])/4. - (n3[i]*n5[i])/4. + (n1[i]*n6[i])/2. + (n2[i]*n6[i])/4. - (n2[i]*n7[i])/2. - n3[i]*n7[i] - (n5[i]*n7[i])/4. - (n6[i]*n7[i])/2. + (n1[i]*n8[i])/2. - (n3[i]*n8[i])/2. + (n4[i]*n8[i])/4. - (n6[i]*n8[i])/4. + n1[i]*n9[i] + (n2[i]*n9[i])/2. + (n4[i]*n9[i])/2. + (n5[i]*n9[i])/4.)
        denom[i] = 1.*n[i]*(n[i]-1)
    return 2. * numer / denom     ### check factor of four

cpdef compute_D2(np.ndarray[np.int32_t, ndim=2] Counts):
    """
    D2 for block of genotypes counted from count_genotypes(Gs)
    """
    cdef np.ndarray[np.int32_t, ndim=1] n1 = Counts[:,0]
    cdef np.ndarray[np.int32_t, ndim=1] n2 = Counts[:,1]
    cdef np.ndarray[np.int32_t, ndim=1] n3 = Counts[:,2]
    cdef np.ndarray[np.int32_t, ndim=1] n4 = Counts[:,3]
    cdef np.ndarray[np.int32_t, ndim=1] n5 = Counts[:,4]
    cdef np.ndarray[np.int32_t, ndim=1] n6 = Counts[:,5]
    cdef np.ndarray[np.int32_t, ndim=1] n7 = Counts[:,6]
    cdef np.ndarray[np.int32_t, ndim=1] n8 = Counts[:,7]
    cdef np.ndarray[np.int32_t, ndim=1] n9 = Counts[:,8]
    cdef np.ndarray[np.int64_t, ndim=1] n
    n = np.sum(Counts, axis=1)
    cdef np.ndarray[np.float_t, ndim=1] numer
    cdef np.ndarray[np.float_t, ndim=1] denom

    cdef int i

    denom = 0.*n
    numer = 0.*n
    for i in range(len(n)):
        numer[i] = (n2[i]*n4[i] - n2[i]**2*n4[i] + 4*n3[i]*n4[i] - 4*n2[i]*n3[i]*n4[i] - 4*n3[i]**2*n4[i] - n2[i]*n4[i]**2 - 4*n3[i]*n4[i]**2 + n1[i]*n5[i] - n1[i]**2*n5[i] + n3[i]*n5[i] + 2*n1[i]*n3[i]*n5[i] - n3[i]**2*n5[i] - 4*n3[i]*n4[i]*n5[i] - n1[i]*n5[i]**2 - n3[i]*n5[i]**2 + 4*n1[i]*n6[i] - 4*n1[i]**2*n6[i] + n2[i]*n6[i] - 4*n1[i]*n2[i]*n6[i] - n2[i]**2*n6[i] + 2*n2[i]*n4[i]*n6[i] - 4*n1[i]*n5[i]*n6[i] - 4*n1[i]*n6[i]**2 - n2[i]*n6[i]**2 + 4*n2[i]*n7[i] - 4*n2[i]**2*n7[i] + 16*n3[i]*n7[i] - 16*n2[i]*n3[i]*n7[i] - 16*n3[i]**2*n7[i] - 4*n2[i]*n4[i]*n7[i] - 16*n3[i]*n4[i]*n7[i] + n5[i]*n7[i] + 2*n1[i]*n5[i]*n7[i] - 4*n2[i]*n5[i]*n7[i] - 18*n3[i]*n5[i]*n7[i] - n5[i]**2*n7[i] + 4*n6[i]*n7[i] + 8*n1[i]*n6[i]*n7[i] - 16*n3[i]*n6[i]*n7[i] - 4*n5[i]*n6[i]*n7[i] - 4*n6[i]**2*n7[i] - 4*n2[i]*n7[i]**2 - 16*n3[i]*n7[i]**2 - n5[i]*n7[i]**2 - 4*n6[i]*n7[i]**2 + 4*n1[i]*n8[i] - 4*n1[i]**2*n8[i] + 4*n3[i]*n8[i] + 8*n1[i]*n3[i]*n8[i] - 4*n3[i]**2*n8[i] + n4[i]*n8[i] - 4*n1[i]*n4[i]*n8[i] + 2*n2[i]*n4[i]*n8[i] - n4[i]**2*n8[i] - 4*n1[i]*n5[i]*n8[i] - 4*n3[i]*n5[i]*n8[i] + n6[i]*n8[i] + 2*n2[i]*n6[i]*n8[i] - 4*n3[i]*n6[i]*n8[i] + 2*n4[i]*n6[i]*n8[i] - n6[i]**2*n8[i] - 16*n3[i]*n7[i]*n8[i] - 4*n6[i]*n7[i]*n8[i] - 4*n1[i]*n8[i]**2 - 4*n3[i]*n8[i]**2 - n4[i]*n8[i]**2 - n6[i]*n8[i]**2 + 16*n1[i]*n9[i] - 16*n1[i]**2*n9[i] + 4*n2[i]*n9[i] - 16*n1[i]*n2[i]*n9[i] - 4*n2[i]**2*n9[i] + 4*n4[i]*n9[i] - 16*n1[i]*n4[i]*n9[i] + 8*n3[i]*n4[i]*n9[i] - 4*n4[i]**2*n9[i] + n5[i]*n9[i] - 18*n1[i]*n5[i]*n9[i] - 4*n2[i]*n5[i]*n9[i] + 2*n3[i]*n5[i]*n9[i] - 4*n4[i]*n5[i]*n9[i] - n5[i]**2*n9[i] - 16*n1[i]*n6[i]*n9[i] - 4*n2[i]*n6[i]*n9[i] + 8*n2[i]*n7[i]*n9[i] + 2*n5[i]*n7[i]*n9[i] - 16*n1[i]*n8[i]*n9[i] - 4*n4[i]*n8[i]*n9[i] - 16*n1[i]*n9[i]**2 - 4*n2[i]*n9[i]**2 - 4*n4[i]*n9[i]**2 - n5[i]*n9[i]**2)/16. + (-((n2[i]/2. + n3[i] + n5[i]/4. + n6[i]/2.)*(n4[i]/2. + n5[i]/4. + n7[i] + n8[i]/2.)) + (n1[i] + n2[i]/2. + n4[i]/2. + n5[i]/4.)*(n5[i]/4. + n6[i]/2. + n8[i]/2. + n9[i]))**2
        denom[i] = 1.*n[i]*(n[i]-1)*(n[i]-2)*(n[i]-3)
    return 4. * numer / denom     ### check factor of four

cpdef compute_Dz(np.ndarray[np.int32_t, ndim=2] Counts):
    """
    Dz for block of genotypes counted from count_genotypes(Gs)
    """
    cdef np.ndarray[np.int32_t, ndim=1] n1 = Counts[:,0]
    cdef np.ndarray[np.int32_t, ndim=1] n2 = Counts[:,1]
    cdef np.ndarray[np.int32_t, ndim=1] n3 = Counts[:,2]
    cdef np.ndarray[np.int32_t, ndim=1] n4 = Counts[:,3]
    cdef np.ndarray[np.int32_t, ndim=1] n5 = Counts[:,4]
    cdef np.ndarray[np.int32_t, ndim=1] n6 = Counts[:,5]
    cdef np.ndarray[np.int32_t, ndim=1] n7 = Counts[:,6]
    cdef np.ndarray[np.int32_t, ndim=1] n8 = Counts[:,7]
    cdef np.ndarray[np.int32_t, ndim=1] n9 = Counts[:,8]
    cdef np.ndarray[np.int64_t, ndim=1] n
    n = np.sum(Counts, axis=1)
    cdef np.ndarray[np.float_t, ndim=1] numer
    cdef np.ndarray[np.float_t, ndim=1] denom

    cdef int i

    denom = 0.*n
    numer = 0.*n
    for i in range(len(n)):
        numer[i] = (-(n2[i]*n4[i]) + 3*n1[i]*n2[i]*n4[i] + n2[i]**2*n4[i] + 2*n3[i]*n4[i] + 4*n1[i]*n3[i]*n4[i] - n2[i]*n3[i]*n4[i] - 4*n3[i]**2*n4[i] + n2[i]*n4[i]**2 + 2*n3[i]*n4[i]**2 + 2*n1[i]*n5[i] - 3*n1[i]**2*n5[i] - n1[i]*n2[i]*n5[i] + 2*n3[i]*n5[i] + 2*n1[i]*n3[i]*n5[i] - n2[i]*n3[i]*n5[i] - 3*n3[i]**2*n5[i] - n1[i]*n4[i]*n5[i] + n3[i]*n4[i]*n5[i] + 2*n1[i]*n6[i] - 4*n1[i]**2*n6[i] - n2[i]*n6[i] - n1[i]*n2[i]*n6[i] + n2[i]**2*n6[i] + 4*n1[i]*n3[i]*n6[i] + 3*n2[i]*n3[i]*n6[i] - 2*n1[i]*n4[i]*n6[i] - 2*n2[i]*n4[i]*n6[i] - 2*n3[i]*n4[i]*n6[i] + n1[i]*n5[i]*n6[i] - n3[i]*n5[i]*n6[i] + 2*n1[i]*n6[i]**2 + n2[i]*n6[i]**2 + 2*n2[i]*n7[i] + 4*n1[i]*n2[i]*n7[i] + 2*n2[i]**2*n7[i] + 8*n3[i]*n7[i] + 4*n1[i]*n3[i]*n7[i] - 4*n3[i]**2*n7[i] - n2[i]*n4[i]*n7[i] + 2*n5[i]*n7[i] + 2*n1[i]*n5[i]*n7[i] + n2[i]*n5[i]*n7[i] + 2*n3[i]*n5[i]*n7[i] - n4[i]*n5[i]*n7[i] + 2*n6[i]*n7[i] - n2[i]*n6[i]*n7[i] - 2*n4[i]*n6[i]*n7[i] + n5[i]*n6[i]*n7[i] + 2*n6[i]**2*n7[i] - 4*n2[i]*n7[i]**2 - 4*n3[i]*n7[i]**2 - 3*n5[i]*n7[i]**2 - 4*n6[i]*n7[i]**2 + 2*n1[i]*n8[i] - 4*n1[i]**2*n8[i] - 2*n1[i]*n2[i]*n8[i] + 2*n3[i]*n8[i] - 2*n2[i]*n3[i]*n8[i] - 4*n3[i]**2*n8[i] - n4[i]*n8[i] - n1[i]*n4[i]*n8[i] - 2*n2[i]*n4[i]*n8[i] - n3[i]*n4[i]*n8[i] + n4[i]**2*n8[i] + n1[i]*n5[i]*n8[i] + n3[i]*n5[i]*n8[i] - n6[i]*n8[i] - n1[i]*n6[i]*n8[i] - 2*n2[i]*n6[i]*n8[i] - n3[i]*n6[i]*n8[i] - 2*n4[i]*n6[i]*n8[i] + n6[i]**2*n8[i] + 4*n1[i]*n7[i]*n8[i] - 2*n2[i]*n7[i]*n8[i] + 3*n4[i]*n7[i]*n8[i] - n5[i]*n7[i]*n8[i] - n6[i]*n7[i]*n8[i] + 2*n1[i]*n8[i]**2 + 2*n3[i]*n8[i]**2 + n4[i]*n8[i]**2 + n6[i]*n8[i]**2 + 8*n1[i]*n9[i] - 4*n1[i]**2*n9[i] + 2*n2[i]*n9[i] + 2*n2[i]**2*n9[i] + 4*n1[i]*n3[i]*n9[i] + 4*n2[i]*n3[i]*n9[i] + 2*n4[i]*n9[i] - n2[i]*n4[i]*n9[i] + 2*n4[i]**2*n9[i] + 2*n5[i]*n9[i] + 2*n1[i]*n5[i]*n9[i] + n2[i]*n5[i]*n9[i] + 2*n3[i]*n5[i]*n9[i] + n4[i]*n5[i]*n9[i] - n2[i]*n6[i]*n9[i] - 2*n4[i]*n6[i]*n9[i] - n5[i]*n6[i]*n9[i] + 4*n1[i]*n7[i]*n9[i] + 4*n3[i]*n7[i]*n9[i] + 4*n4[i]*n7[i]*n9[i] + 2*n5[i]*n7[i]*n9[i] + 4*n6[i]*n7[i]*n9[i] - 2*n2[i]*n8[i]*n9[i] + 4*n3[i]*n8[i]*n9[i] - n4[i]*n8[i]*n9[i] - n5[i]*n8[i]*n9[i] + 3*n6[i]*n8[i]*n9[i] - 4*n1[i]*n9[i]**2 - 4*n2[i]*n9[i]**2 - 4*n4[i]*n9[i]**2 - 3*n5[i]*n9[i]**2)/4. + (-n1[i] + n3[i] - n4[i] + n6[i] - n7[i] + n9[i])*(-n1[i] - n2[i] - n3[i] + n7[i] + n8[i] + n9[i])*(-((n2[i]/2. + n3[i] + n5[i]/4. + n6[i]/2.)*(n4[i]/2. + n5[i]/4. + n7[i] + n8[i]/2.)) + (n1[i] + n2[i]/2. + n4[i]/2. + n5[i]/4.)*(n5[i]/4. + n6[i]/2. + n8[i]/2. + n9[i]))
        denom[i] = 1.*n[i]*(n[i]-1)*(n[i]-2)*(n[i]-3)
    return 2. * numer / denom     ### check factor of four

cpdef compute_pi2(np.ndarray[np.int32_t, ndim=2] Counts):
    """
    pi2 for block of genotypes counted from count_genotypes(Gs)
    """
    cdef np.ndarray[np.int32_t, ndim=1] n1 = Counts[:,0]
    cdef np.ndarray[np.int32_t, ndim=1] n2 = Counts[:,1]
    cdef np.ndarray[np.int32_t, ndim=1] n3 = Counts[:,2]
    cdef np.ndarray[np.int32_t, ndim=1] n4 = Counts[:,3]
    cdef np.ndarray[np.int32_t, ndim=1] n5 = Counts[:,4]
    cdef np.ndarray[np.int32_t, ndim=1] n6 = Counts[:,5]
    cdef np.ndarray[np.int32_t, ndim=1] n7 = Counts[:,6]
    cdef np.ndarray[np.int32_t, ndim=1] n8 = Counts[:,7]
    cdef np.ndarray[np.int32_t, ndim=1] n9 = Counts[:,8]
    cdef np.ndarray[np.int64_t, ndim=1] n
    n = np.sum(Counts, axis=1)
    cdef np.ndarray[np.float_t, ndim=1] numer
    cdef np.ndarray[np.float_t, ndim=1] denom

    cdef int i

    denom = 0.*n
    numer = 0.*n
    for i in range(len(n)):
        numer[i] = ((n1[i] + n2[i] + n3[i] + n4[i]/2. + n5[i]/2. + n6[i]/2.)*(n1[i] + n2[i]/2. + n4[i] + n5[i]/2. + n7[i] + n8[i]/2.) *
                    (n2[i]/2. + n3[i] + n5[i]/2. + n6[i] + n8[i]/2. + n9[i])*(n4[i]/2. + n5[i]/2. + n6[i]/2. + n7[i] + n8[i] + n9[i]) + 
                    ((13*n2[i]*n4[i] - 16*n1[i]*n2[i]*n4[i] - 11*n2[i]**2*n4[i] + 16*n3[i]*n4[i] - 28*n1[i]*n3[i]*n4[i] - 24*n2[i]*n3[i]*n4[i]) + 
                    (-8*n3[i]**2*n4[i] - 11*n2[i]*n4[i]**2 - 20*n3[i]*n4[i]**2 - 6*n5[i] + 12*n1[i]*n5[i] - 4*n1[i]**2*n5[i] + 17*n2[i]*n5[i]) +
                    (-20*n1[i]*n2[i]*n5[i] - 11*n2[i]**2*n5[i] + 12*n3[i]*n5[i] - 28*n1[i]*n3[i]*n5[i] - 20*n2[i]*n3[i]*n5[i] - 4*n3[i]**2*n5[i]) + 
                    (17*n4[i]*n5[i] - 20*n1[i]*n4[i]*n5[i] - 32*n2[i]*n4[i]*n5[i] - 40*n3[i]*n4[i]*n5[i] - 11*n4[i]**2*n5[i] + 11*n5[i]**2 - 16*n1[i]*n5[i]**2) + 
                    (-17*n2[i]*n5[i]**2 - 16*n3[i]*n5[i]**2 - 17*n4[i]*n5[i]**2 - 6*n5[i]**3 + 16*n1[i]*n6[i] - 8*n1[i]**2*n6[i] + 13*n2[i]*n6[i] - 24*n1[i]*n2[i]*n6[i]) + 
                    (-11*n2[i]**2*n6[i] - 28*n1[i]*n3[i]*n6[i] - 16*n2[i]*n3[i]*n6[i] + 24*n4[i]*n6[i] - 36*n1[i]*n4[i]*n6[i] - 38*n2[i]*n4[i]*n6[i] - 36*n3[i]*n4[i]*n6[i]) + 
                    (-20*n4[i]**2*n6[i] + 17*n5[i]*n6[i] - 40*n1[i]*n5[i]*n6[i] - 32*n2[i]*n5[i]*n6[i] - 20*n3[i]*n5[i]*n6[i] - 42*n4[i]*n5[i]*n6[i] - 17*n5[i]**2*n6[i]) + 
                    (-20*n1[i]*n6[i]**2 - 11*n2[i]*n6[i]**2 - 20*n4[i]*n6[i]**2 - 11*n5[i]*n6[i]**2 + 16*n2[i]*n7[i] - 28*n1[i]*n2[i]*n7[i] - 20*n2[i]**2*n7[i] + 16*n3[i]*n7[i]) + 
                    (-48*n1[i]*n3[i]*n7[i] - 44*n2[i]*n3[i]*n7[i] - 16*n3[i]**2*n7[i] - 24*n2[i]*n4[i]*n7[i] - 44*n3[i]*n4[i]*n7[i] + 12*n5[i]*n7[i] - 28*n1[i]*n5[i]*n7[i]) + 
                    (-40*n2[i]*n5[i]*n7[i] - 48*n3[i]*n5[i]*n7[i] - 20*n4[i]*n5[i]*n7[i] - 16*n5[i]**2*n7[i] + 16*n6[i]*n7[i] - 48*n1[i]*n6[i]*n7[i] - 48*n2[i]*n6[i]*n7[i]) + 
                    (-44*n3[i]*n6[i]*n7[i] - 36*n4[i]*n6[i]*n7[i] - 40*n5[i]*n6[i]*n7[i] - 20*n6[i]**2*n7[i] - 8*n2[i]*n7[i]**2 - 16*n3[i]*n7[i]**2 - 4*n5[i]*n7[i]**2) + 
                    (-8*n6[i]*n7[i]**2 + 16*n1[i]*n8[i] - 8*n1[i]**2*n8[i] + 24*n2[i]*n8[i] - 36*n1[i]*n2[i]*n8[i] - 20*n2[i]**2*n8[i] + 16*n3[i]*n8[i] - 48*n1[i]*n3[i]*n8[i]) + 
                    (-36*n2[i]*n3[i]*n8[i] - 8*n3[i]**2*n8[i] + 13*n4[i]*n8[i] - 24*n1[i]*n4[i]*n8[i] - 38*n2[i]*n4[i]*n8[i] - 48*n3[i]*n4[i]*n8[i] - 11*n4[i]**2*n8[i]) + 
                    (17*n5[i]*n8[i] - 40*n1[i]*n5[i]*n8[i] - 42*n2[i]*n5[i]*n8[i] - 40*n3[i]*n5[i]*n8[i] - 32*n4[i]*n5[i]*n8[i] - 17*n5[i]**2*n8[i] + 13*n6[i]*n8[i]) + 
                    (-48*n1[i]*n6[i]*n8[i] - 38*n2[i]*n6[i]*n8[i] - 24*n3[i]*n6[i]*n8[i] - 38*n4[i]*n6[i]*n8[i] - 32*n5[i]*n6[i]*n8[i] - 11*n6[i]**2*n8[i] - 28*n1[i]*n7[i]*n8[i]) + 
                    (-36*n2[i]*n7[i]*n8[i] - 44*n3[i]*n7[i]*n8[i] - 16*n4[i]*n7[i]*n8[i] - 20*n5[i]*n7[i]*n8[i] - 24*n6[i]*n7[i]*n8[i] - 20*n1[i]*n8[i]**2 - 20*n2[i]*n8[i]**2) + 
                    (-20*n3[i]*n8[i]**2 - 11*n4[i]*n8[i]**2 - 11*n5[i]*n8[i]**2 - 11*n6[i]*n8[i]**2 + 16*n1[i]*n9[i] - 16*n1[i]**2*n9[i] + 16*n2[i]*n9[i] - 44*n1[i]*n2[i]*n9[i]) + 
                    (-20*n2[i]**2*n9[i] - 48*n1[i]*n3[i]*n9[i] - 28*n2[i]*n3[i]*n9[i] + 16*n4[i]*n9[i] - 44*n1[i]*n4[i]*n9[i] - 48*n2[i]*n4[i]*n9[i] - 48*n3[i]*n4[i]*n9[i]) + 
                    (-20*n4[i]**2*n9[i] + 12*n5[i]*n9[i] - 48*n1[i]*n5[i]*n9[i] - 40*n2[i]*n5[i]*n9[i] - 28*n3[i]*n5[i]*n9[i] - 40*n4[i]*n5[i]*n9[i] - 16*n5[i]**2*n9[i]) + 
                    (-44*n1[i]*n6[i]*n9[i] - 24*n2[i]*n6[i]*n9[i] - 36*n4[i]*n6[i]*n9[i] - 20*n5[i]*n6[i]*n9[i] - 48*n1[i]*n7[i]*n9[i] - 48*n2[i]*n7[i]*n9[i]) + 
                    (-48*n3[i]*n7[i]*n9[i] - 28*n4[i]*n7[i]*n9[i] - 28*n5[i]*n7[i]*n9[i] - 28*n6[i]*n7[i]*n9[i] - 44*n1[i]*n8[i]*n9[i] - 36*n2[i]*n8[i]*n9[i]) + 
                    (-28*n3[i]*n8[i]*n9[i] - 24*n4[i]*n8[i]*n9[i] - 20*n5[i]*n8[i]*n9[i] - 16*n6[i]*n8[i]*n9[i] - 16*n1[i]*n9[i]**2 - 8*n2[i]*n9[i]**2 - 8*n4[i]*n9[i]**2 - 4*n5[i]*n9[i]**2))/16.)
        denom[i] = 1.*n[i]*(n[i]-1)*(n[i]-2)*(n[i]-3)
    return numer / denom
