import numpy
import scipy.misc as misc

import Spectrum_mod

"""
Usefull functions for Spectra manipulations:
"""

# population splits

# to be checked: this method can be used for nD to n+1D splits just applying it to a well chosen sub-spectrum of sp...
def split_1D_to_2D(sp, n1, n2):
    """
    One-to-two population split for the spectrum.
    n1, n2 are population sizes for the two resulting populations
    needs the spectrum to be 1D and n >= n1+n2
    Returns a new 2D spectrum
    """
    assert(len(sp.shape) == 1)
    assert(len(sp) >= n1+n2+1)
    sp.unmask_all()
    # if the sample size before split is too large, we project
    if len(sp) > n1+n2+1:
        sp.project([n1 + n2 + 1])
    # then we compute the joint fs resulting from the split
    data_2D = numpy.zeros((n1+1, n2+1))
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            data_2D[i, j] = sp[i+j] * misc.comb(n1, i) * misc.comb(n2, j) / misc.comb(n1+n2, i+j)
    #data_2D = numpy.array([[sp[i+j]*misc.comb(n1,i)*misc.comb(n2,j)/misc.comb(n1+n2,i+j) for j in range(n2+1)] for i in range(n1+1)])
    return Spectrum_mod.Spectrum(data_2D)

def split_2D_to_3D_2(sp, n2new, n3):
    """
    Two-to-three population split for the spectrum.
    n2new, n3 are population sizes for the two resulting populations
    needs the spectrum to be 2D and n2 >= n2new+n3
    Returns a new 3D spectrum
    """
    assert(len(sp.shape) == 2)
    n1 = sp.shape[0] - 1
    n2 = sp.shape[1] - 1
    assert(n2 >= n2new+n3)
    sp.unmask_all()
    # if the sample size before split is too large, we project
    if n2 > n2new+n3:
        sp.project([n1, n2new+n3+1])
    
    # then we compute the join fs resulting from the split
    data_3D = numpy.zeros((n1+1, n2new+1, n3+1))
    for i in range(n2new + 1):
        for j in range(n3 + 1):
            data_3D[:, i, j] = sp[:, i+j] * misc.comb(n2new, i) * misc.comb(n3, j) / misc.comb(n2new+n3, i+j)
    return Spectrum_mod.Spectrum(data_3D)

def split_2D_to_3D_1(sp, n1new, n3):
    """
    Two-to-three population split for the spectrum.
    n1new, n3 are population sizes for the two resulting populations
    needs the spectrum to be 2D and n1 >= n1new+n3
    Returns a new 3D spectrum
    """
    assert(len(sp.shape) == 2)
    n1 = sp.shape[0]-1
    n2 = sp.shape[1]-1
    assert(n1 >= n1new+n3)
    sp.unmask_all()
    # if the sample size before split is too large, we project
    if n1 > n1new+n3:
        sp.project([n1new+n3+1, n2])
    
    # then we compute the join fs resulting from the split
    data_3D = numpy.zeros((n1new+1, n2+1, n3+1))
    for i in range(n1new + 1):
        for j in range(n3 + 1):
            data_3D[i, :, j] = sp[i+j, :] * misc.comb(n1new, i) * misc.comb(n3, j) / misc.comb(n1new+n3, i+j)
    return Spectrum_mod.Spectrum(data_3D)

# merge two populations into one population
def merge_2D_to_1D(sp):
    """
    Two-to-one populations fusion
    needs the input spectrum to be 2D
    Returns a new 1D spectrum
    """
    assert(len(sp.shape) == 2)
    sp.unmask_all()
    dim1, dim2 = sp.shape
    data = numpy.zeros(dim1 + dim2 - 1)
    for k in range(dim1):
        for l in range(dim2):
            data[k + l] += sp[k, l]
    return Spectrum_mod.Spectrum(data)
