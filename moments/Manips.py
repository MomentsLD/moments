import numpy as np
import scipy.misc as misc

import ModelPlot
import Spectrum_mod
from scipy.optimize import _nnls
import scipy as sp
from numpy import asarray_chkfinite, zeros, double

"""
Usefull functions for Spectra manipulations:
"""

# population splits

def split_1D_to_2D(sp, n1, n2):
    """
    One-to-two population split for the spectrum,
    needs that n >= n1+n2.

    sp : 1D spectrum
    
    n1 : sample size for resulting pop 1
    
    n2 : sample size for resulting pop 2
    
    Returns a new 2D spectrum
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(0, (0,1))
    
    assert(len(sp.shape) == 1)
    assert(len(sp) >= n1 + n2 + 1)
    # if the sample size before split is too large, we project
    if len(sp) > n1 + n2 + 1:
        sp = sp.project([n1 + n2 + 1])
    sp.unmask_all()
    
    # then we compute the joint fs resulting from the split
    data_2D = np.zeros((n1 + 1, n2 + 1))
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            data_2D[i, j] = sp[i + j] * misc.comb(n1, i) * misc.comb(n2, j)  \
                            / misc.comb(n1 + n2, i + j)
    return Spectrum_mod.Spectrum(data_2D)

def split_2D_to_3D_2(sp, n2new, n3):
    """
    Two-to-three population split for the spectrum, 
    needs that n2 >= n2new+n3.

    sp : 2D spectrum

    n2new : sample size for resulting pop 2

    n3 : sample size for resulting pop 3

    Returns a new 3D spectrum
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(1, (1,2))
    
    assert(len(sp.shape) == 2)
    n1 = sp.shape[0] - 1
    n2 = sp.shape[1] - 1
    assert(n2 >= n2new + n3)
    # if the sample size before split is too large, we project
    if n2 > n2new + n3:
        sp = sp.project([n1, n2new + n3 + 1])
    sp.unmask_all()
    
    # then we compute the join fs resulting from the split
    data_3D = np.zeros((n1 + 1, n2new + 1, n3 + 1))
    for i in range(n2new + 1):
        for j in range(n3 + 1):
            data_3D[:, i, j] = sp[:, i + j] * misc.comb(n2new, i) * misc.comb(n3, j)  \
                               / misc.comb(n2new + n3, i + j)
    return Spectrum_mod.Spectrum(data_3D)

def split_2D_to_3D_1(sp, n1new, n3):
    """
    Two-to-three population split for the spectrum, 
    needs that n2 >= n2new+n3.

    sp : 2D spectrum
    
    n1new : sample size for resulting pop 1

    n3 : sample size for resulting pop 3
    
    Returns a new 3D spectrum
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(0, (0,2))
  
    assert(len(sp.shape) == 2)
    n1 = sp.shape[0] - 1
    n2 = sp.shape[1] - 1
    assert(n1 >= n1new + n3)
    # if the sample size before split is too large, we project
    if n1 > n1new + n3:
        sp = sp.project([n1new + n3 + 1, n2])
    sp.unmask_all()
    
    # then we compute the join fs resulting from the split
    data_3D = np.zeros((n1new + 1, n2 + 1, n3 + 1))
    for i in range(n1new + 1):
        for j in range(n3 + 1):
            data_3D[i, :, j] = sp[i + j, :] * misc.comb(n1new, i) * misc.comb(n3, j)  \
                               / misc.comb(n1new + n3, i + j)
    return Spectrum_mod.Spectrum(data_3D)


def split_3D_to_4D_3(sp, n3new, n4):
    """
    Three-to-four population split for the spectrum,
    needs that n3 >= n3new+n4.

    sp : 3D spectrum

    n3new : sample size for resulting pop 3

    n4 : sample size for resulting pop 4 
   
    Returns a new 4D spectrum
    """ 
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(2, (2,3))
  
    assert(len(sp.shape) == 3)
    n1 = sp.shape[0] - 1
    n2 = sp.shape[1] - 1
    n3 = sp.shape[2] - 1
    assert(n3 >= n3new + n4)
    # if the sample size before split is too large, we project
    if n3 > n3new + n4:
        sp = sp.project([n1, n2, n3new + n4 + 1])
    sp.unmask_all()
    
    # then we compute the join fs resulting from the split
    data_4D = np.zeros((n1 + 1, n2 + 1, n3new + 1, n4 + 1))
    for i in range(n3new + 1):
        for j in range(n4 + 1):
            data_4D[:, :, i, j] = sp[:, :, i + j] * misc.comb(n3new, i) * misc.comb(n4, j)  \
                                  / misc.comb(n3new + n4, i + j)
    return Spectrum_mod.Spectrum(data_4D)

def split_4D_to_5D_4(sp, n4new, n5):
    """
    Four-to-five population split for the spectrum,
    n4 >= n4new+n5.

    sp : 4D spectrum
    
    n4new : sample size for resulting pop 4

    n5 : sample size for resulting pop 5
    
    Returns a new 5D spectrum
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(3, (3,4))
    
    assert(len(sp.shape) == 4)
    n1 = sp.shape[0] - 1
    n2 = sp.shape[1] - 1
    n3 = sp.shape[2] - 1
    n4 = sp.shape[3] - 1
    assert(n4 >= n4new + n5)
    # if the sample size before split is too large, we project
    if n4 > n4new + n5:
        sp = sp.project([n1, n2, n3, n4new + n5 + 1])
    sp.unmask_all()
    
    # then we compute the join fs resulting from the split
    data_5D = np.zeros((n1 + 1, n2 + 1, n3 + 1, n4new + 1, n5 + 1))
    for i in range(n4new + 1):
        for j in range(n5 + 1):
            data_5D[:, :, :, i, j] = sp[:, :, :, i + j] * misc.comb(n4new, i)  \
                                     * misc.comb(n5, j) / misc.comb(n4new + n5, i + j)
    return Spectrum_mod.Spectrum(data_5D)

def split_4D_to_5D_3(sp, n3new, n4):
    """
    Four-to-five population split for the spectrum,
    n3 >= n3new+n4.

    sp : 4D spectrum
    
    n3new : sample size for resulting pop 3

    n4 : sample size for resulting pop 4
    
    Returns a new 5D spectrum
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(2, (2,3))
    
    assert(len(sp.shape) == 4)
    n1 = sp.shape[0] - 1
    n2 = sp.shape[1] - 1
    n3 = sp.shape[2] - 1
    n5 = sp.shape[3] - 1
    assert(n3 >= n3new + n4)
    # if the sample size before split is too large, we project
    if n3 > n3new + n4:
        sp = sp.project([n1, n2, n3new + n4 + 1, n5])
    sp.unmask_all()
    
    # then we compute the join fs resulting from the split
    data_5D = np.zeros((n1 + 1, n2 + 1, n3new + 1, n4 + 1, n5 + 1))
    for i in range(n3new + 1):
        for j in range(n4 + 1):
            data_5D[:, :, i, j, :] = sp[:, :, i + j, :] * misc.comb(n3new, i)  \
                                     * misc.comb(n4, j) / misc.comb(n3new + n4, i + j)
    return Spectrum_mod.Spectrum(data_5D)


# merge two populations into one population
def merge_2D_to_1D(sp):
    """
    Two-to-one populations fusion
    
    sp : 2D spectrum
    
    Returns a new 1D spectrum
    """
    assert(len(sp.shape) == 2)
    sp.unmask_all()
    dim1, dim2 = sp.shape
    data = np.zeros(dim1 + dim2 - 1)
    for k in range(dim1):
        for l in range(dim2):
            data[k + l] += sp[k, l]
    return Spectrum_mod.Spectrum(data)


#  Methods for admixture

def __drop_last_slice__(sfs, dimension):
    #drop last slice along dimension in sfs
    
    ns = sfs.shape
    dim = len(ns)
    if dimension<0:
        dimension = dim + dimension
    slicing = (slice(None),) * dimension  + (slice(None,-1),) + (slice(None),) * (dim-1-dimension)
    return slicing
    
def __drop_first_slice__(sfs, dimension):
    #drop first slice along dimension in sfs
    ns = sfs.shape
    dim = len(ns)
    if dimension<0:
        dimension = dim + dimension
    slicing = (slice(None),) * dimension  + (slice(1,None),) + (slice(None),) * (dim-1-dimension)
    return slicing

def __migrate_1__(sfs, dimension1, dimension2):
    """Takes SFS , pick one individual from population dimension1 and migrate it to 
    population dimension2. If sfs has dimension (m,n), the new sfs will have dimension 
    (m-1,n+1)"""

    ns = sfs.shape
    new_ns = list(ns)
    M = ns[dimension1]-1
    N = ns[dimension2]-1
    
    new_ns[dimension1] -= 1
    new_ns[dimension2] += 1
    new_sfs = Spectrum_mod.Spectrum(np.zeros(new_ns))
    
    # We first suppose that we pick a reference allele. 
    
    # since we picked a reference allele, there can be no contribution from the
    # sfs[:,:,ns[dimension1],:,:], which would have all alts
        
    new_sfs[__drop_last_slice__(new_sfs,dimension2)]\
                = (sfs[__drop_last_slice__(sfs,dimension1)].swapaxes(dimension1,-1)\
                    * (1 - np.arange(M)*1./M)).swapaxes(dimension1,-1)
                    
    
    new_sfs[__drop_first_slice__(new_sfs,dimension2)]\
                += (sfs[__drop_first_slice__(sfs,dimension1)].swapaxes(dimension1,-1)\
                    * (np.arange(1,M+1)*1./M)).swapaxes(dimension1,-1)                 

    return new_sfs


def __nnls_mod__(A, b):
    """
    SG: I modified the scipy.optimize.nnls function to return the best-found parameters 
    even if the nnls algorithm has not converged, and issue a warning rather than crash.  
    The instructions below are from the original function
    
    
    Solve ``argmin_x || Ax - b ||_2`` for ``x>=0``. This is a wrapper
    for a FORTAN non-negative least squares solver.

    Parameters
    ----------
    A : ndarray
        Matrix ``A`` as shown above.
    b : ndarray
        Right-hand side vector.

    Returns
    -------
    x : ndarray
        Solution vector.
    rnorm : float
        The residual, ``|| Ax-b ||_2``.

    Notes
    -----
    The FORTRAN code was published in the book below. The algorithm
    is an active set method. It solves the KKT (Karush-Kuhn-Tucker)
    conditions for the non-negative least squares problem.

    References
    ----------
    Lawson C., Hanson R.J., (1987) Solving Least Squares Problems, SIAM

    """

    A, b = map(asarray_chkfinite, (A, b))

    if len(A.shape) != 2:
        raise ValueError("expected matrix")
    if len(b.shape) != 1:
        raise ValueError("expected vector")

    m, n = A.shape

    if m != b.shape[0]:
        raise ValueError("incompatible dimensions")

    w = zeros((n,), dtype=double)
    zz = zeros((m,), dtype=double)
    index = zeros((n,), dtype=int)

    x, rnorm, mode = _nnls.nnls(A, m, n, b, w, zz, index)
    if mode != 1:
        print("Warning: too many iterations in nnls") #SG my modification

    return x, rnorm

def __Gamma__(n_draws,n_lineages):
    """ The gamma matrix element i,j gives the probability that a sequential sample of i 
    lineages with replacement gives j distinct lineages
    """
    # the first row is the probability that a sample of 0 lineages gives j distinct 
    # lineages: it is always 0 distinct lineages
    current = np.zeros(n_lineages+1)
    current[0] = 1
    # then we compute the other rows through dynamic programming, adding one other sample
    # at a time: if we have a sample of size n, there are two possibilities for sample of 
    # size n+1: either we draw an existing allele, or we draw a new allele. 
    
    # then we compute the other rows through dynamic programming, adding one other sample
    # at a time: if had drawn n replacements, there are two possibilities for the n+1th
    # replacement: either we draw an existing allele, or we draw a new allele. 
    
    list_arrays = [current]
    transition_matrix = np.diag([i*1./n_lineages for i in range(n_lineages+1)])\
                        + np.diag([1-i*1./n_lineages for i in range(n_lineages)], k=-1)
    
    for i in range(n_draws):
        list_arrays.append(np.dot(transition_matrix,list_arrays[-1]))
    return np.array(list_arrays)

# Admixture of population 1 and 2 into a new population [-1], using the exact dp approach 

def admix_into_new(sfs, dimension1, dimension2, n_lineages, m1):
    """
    creates n_lineages in a new dimension to the SFS by drawing each from
    pops dimension1 (with probability m1) and dimension2 (with probability 1-m1).  
    
    The resulting frequency spectrum has 
    
    sfs a frequency spectrum
    dimension1: label of population 1
    dimension2: label of population 2
    m1 proportion of lineages drawn from pop 1
    creates a lst dimension in which to insert the new population
    """
    dimensions = sfs.shape
    new_dimensions = list(dimensions)+[1] 
    M = dimensions[dimension1]-1
    N = dimensions[dimension2]-1
    new_sfs = sfs.reshape(new_dimensions)
    
    assert n_lineages <= min(M,N), "not enough lineages to produce %d, M=%d,N=%d"\
                                                                     % (n_lineages, M, N)
    project_dimensions = [n-1 for n in new_dimensions] # projection use number of lineages
    
   
    
    for _i in range(n_lineages):
        project_dimensions[-1] += 1
        project_dimensions[dimension1] -= 1
        project_dimensions[dimension2] -= 1
        #print "pd", project_dimensions
        #print (m1 * migrate_1(new_sfs, dimension1,-1)).shape
        #print ((1-m1) * migrate_1(new_sfs, dimension2,-1)).shape
        new_sfs = Spectrum_mod.Spectrum.project(m1 * __migrate_1__(new_sfs, dimension1,-1), 
                                            project_dimensions)\
                +Spectrum_mod.Spectrum.project((1-m1) * __migrate_1__(new_sfs, dimension2,-1),
                                         project_dimensions)
    return np.squeeze(new_sfs) # Remove empty dimensions


# Approximate admixture model

def admix_inplace(sfs, dimension1, dimension2, keep_1, m1):
    """admixes from population1 to population 2 in place, sending migrants one by one, 
    and normalizing so that in the end we have approximately the correct distribution of 
    replaced lineages. 
    
    dimension1: label of population 1
    dimension2: label of population 2
    m1 proportion of lineages in 2 drawn from pop 1
    keep_1: number of lineages from population 1 that we want to keep.
    """
    dimensions = sfs.shape
    M = dimensions[dimension1] - 1 # number of haploid samples is size of sfs - 1
    N = dimensions[dimension2] - 1
    
    target_M = keep_1
    target_N = N
    
    target_dimensions = list(np.array(dimensions[:])-1)
    target_dimensions[dimension1] = target_M
    target_dimensions[dimension2] = target_N
    
    assert keep_1 <= M, "Cannot keep more lineages than we started with, keep_1=%d,\
    M=%d" % (n_lineages, keep_1, M)
   
    ############################
    # We first compute the sequence of SFSs we would obtain by migrating individuals
    # sequentially. This will give us a range of distributions, which we will use to 
    # compute the correct distribution below.
    
    
    max_replacements = M - keep_1 
    
    current_sfs = sfs[:]   

    
    list_sfs = [sfs.project(target_dimensions)]  # Remember the SFSs we computed
    list_replacements = [0]  # The number of replacements in the corresponding sfs
    
    
    for num_replacements in range(1,max_replacements+1):
        # The shape of the sfs is (n1+1, n2+1,...). We want to extract 
        # sample sizes (n1,n2,...)
        project_dimensions = [shape_elem-1 for shape_elem in current_sfs.shape] 
        
        project_dimensions[dimension2] -= 1 #  since there is a migrant, 
                                            # only n2-1 lineages from 2 survive
        
        # first remove one sample from population 2, then migrate one from pop 1 to pop 2
        current_sfs = __migrate_1__(Spectrum_mod.Spectrum.project(current_sfs, project_dimensions),
                                    dimension1, dimension2)
        keeper_function = True #  Eventually we may want to only keep a subset -- 
                               #  but don't want to optimize too early. 
        if keeper_function:
            list_sfs.append(current_sfs.project(target_dimensions))
            list_replacements.append(num_replacements)
    
    ##################
    # Now that we have computed the list of SFSs with sequential migrations, we want to
    # use them to compute the correct frequency spectrum


    gamma = __Gamma__(max_replacements, N) # the conversion matrix giving us the num of 
                                       # replacements after 0,1,...,max_replacements 
                                       # replacements    
    target = np.array([sp.stats.binom(N,m1).pmf(i) for i in range(N+1)]) # binomial is 
                            # the standard, but we could use any distribution! 

    weights = __nnls_mod__(gamma.transpose(), target) # find a positive definite set of 
                                                  # parameters that imitates the target  
    if weights[1] > 0.001:
        print "warning, in binomial distribution approximation is %2.3f, consider\
        including more lineages. If more lineages don't resolve the situation,\
        consider using the exact admixture model" % weights[1]    
    # Following could be optimized by making it a dot product  
    new_sfs=0
    for i in range(len(weights[0])):
        new_sfs+=list_sfs[i]*weights[0][i]
    
    return list_sfs, target, weights, new_sfs

