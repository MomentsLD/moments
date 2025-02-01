import numpy as np
import warnings

from . import ModelPlot
from . import Spectrum_mod

try:
    from scipy.optimize import __nnls as _nnls
except ImportError:
    from scipy.optimize import _nnls
import scipy as sp
from scipy import stats
from numpy import asarray_chkfinite, zeros, double
import copy
from scipy.special import gammaln

"""
Usefull functions for Spectra manipulations:
"""

# population splits
# all split functions leave one population in place and put the new
# population at the end of the population index list


def _log_comb(n, k):
    return gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)


def split_1D_to_2D(sfs, n1, n2):
    """
    One-to-two population split for the spectrum,
    needs that n >= n1+n2.

    sfs : 1D spectrum

    n1 : sample size for resulting pop 1

    n2 : sample size for resulting pop 2

    Returns a new 2D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(0, (0, 1))

    data_1D = copy.copy(sfs)  # copy to preserve masking of sp
    if len(data_1D.shape) != 1:
        raise ValueError("Spectrum is not 1D")
    if len(data_1D) < n1 + n2 + 1:
        raise ValueError(f"Sample size is too small to split into {n1} and {n2}")
    # if the sample size before split is too large, we project
    if len(data_1D) > n1 + n2 + 1:
        data_1D = data_1D.project([n1 + n2])
    data_1D.unmask_all()

    # then we compute the joint fs resulting from the split
    data_2D = np.zeros((n1 + 1, n2 + 1))
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            log_entry = _log_comb(n1, i) + _log_comb(n2, j) - _log_comb(n1 + n2, i + j)
            data_2D[i, j] = data_1D.data[i + j] * np.exp(log_entry)

    data_2D = Spectrum_mod.Spectrum(data_2D, mask_corners=False)
    if mask_lost == True:
        data_2D.mask[0, 0] = True
    if mask_fixed == True:
        data_2D.mask[-1, -1] = True
    return data_2D


def split_2D_to_3D_2(sfs, n2new, n3):
    """
    Two-to-three population split for the spectrum,
    needs that n2 >= n2new+n3.

    sfs : 2D spectrum

    n2new : sample size for resulting pop 2

    n3 : sample size for resulting pop 3

    Returns a new 3D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0, 0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1, -1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(1, (1, 2))

    data_2D = copy.copy(sfs)
    if len(data_2D.shape) != 2:
        raise ValueError("Spectrum is not 2D")
    n1 = data_2D.shape[0] - 1
    n2 = data_2D.shape[1] - 1
    if n2 < n2new + n3:
        raise ValueError(f"Sample size is too small to split into {n2new} and {n3}")
    # if the sample size before split is too large, we project
    if n2 > n2new + n3:
        data_2D = data_2D.project([n1, n2new + n3])
    data_2D.unmask_all()

    # then we compute the joint fs resulting from the split
    data_3D = np.zeros((n1 + 1, n2new + 1, n3 + 1))
    for i in range(n2new + 1):
        for j in range(n3 + 1):
            log_entry_weight = (
                _log_comb(n2new, i) + _log_comb(n3, j) - _log_comb(n2new + n3, i + j)
            )
            data_3D[:, i, j] = data_2D.data[:, i + j] * np.exp(log_entry_weight)

    data_3D = Spectrum_mod.Spectrum(data_3D, mask_corners=False)
    if mask_lost == True:
        data_3D.mask[0, 0, 0] = True
    if mask_fixed == True:
        data_3D.mask[-1, -1, -1] = True
    return data_3D


def split_2D_to_3D_1(sfs, n1new, n3):
    """
    Two-to-three population split for the spectrum,
    needs that n2 >= n2new+n3.

    sfs : 2D spectrum

    n1new : sample size for resulting pop 1

    n3 : sample size for resulting pop 3

    Returns a new 3D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0, 0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1, -1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(0, (0, 2))

    data_2D = copy.copy(sfs)
    if len(data_2D.shape) != 2:
        raise ValueError("Spectrum is not 2D")
    n1 = data_2D.shape[0] - 1
    n2 = data_2D.shape[1] - 1
    if n1 < n1new + n3:
        raise ValueError(f"Sample size is to small to split into {n1new} and {n3}")
    # if the sample size before split is too large, we project
    if n1 > n1new + n3:
        data_2D = data_2D.project([n1new + n3, n2])
    data_2D.unmask_all()

    # then we compute the joint fs resulting from the split
    data_3D = np.zeros((n1new + 1, n2 + 1, n3 + 1))
    for i in range(n1new + 1):
        for j in range(n3 + 1):
            log_entry_weight = (
                _log_comb(n1new, i) + _log_comb(n3, j) - _log_comb(n1new + n3, i + j)
            )
            data_3D[i, :, j] = data_2D.data[i + j, :] * np.exp(log_entry_weight)

    data_3D = Spectrum_mod.Spectrum(data_3D, mask_corners=False)
    if mask_lost == True:
        data_3D.mask[0, 0, 0] = True
    if mask_fixed == True:
        data_3D.mask[-1, -1, -1] = True
    return data_3D


def split_3D_to_4D_3(sfs, n3new, n4, model_update=True):
    """
    Three-to-four population split for the spectrum,
    needs that n3 >= n3new+n4.

    sfs : 3D spectrum

    n3new : sample size for resulting pop 3

    n4 : sample size for resulting pop 4

    model_update : update model if set to True. Since we wrap this function to
        split other populations, we update the ModelPlot model outside of this
        function instead, and set it to False in that case.

    Returns a new 4D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0, 0, 0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1, -1, -1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None and model_update is True:
        model.split(2, (2, 3))

    data_3D = copy.copy(sfs)
    if len(data_3D.shape) != 3:
        raise ValueError("SFS is not 3D")
    n1 = data_3D.shape[0] - 1
    n2 = data_3D.shape[1] - 1
    n3 = data_3D.shape[2] - 1
    if n3 < n3new + n4:
        raise ValueError("requested sample sizes are too large")
    # if the sample size before split is too large, we project
    if n3 > n3new + n4:
        data_3D = data_3D.project([n1, n2, n3new + n4])
    data_3D.unmask_all()

    # then we compute the joint fs resulting from the split
    data_4D = np.zeros((n1 + 1, n2 + 1, n3new + 1, n4 + 1))
    for i in range(n3new + 1):
        for j in range(n4 + 1):
            log_entry_weight = (
                _log_comb(n3new, i) + _log_comb(n4, j) - _log_comb(n3new + n4, i + j)
            )
            data_4D[:, :, i, j] = data_3D.data[:, :, i + j] * np.exp(log_entry_weight)

    data_4D = Spectrum_mod.Spectrum(data_4D, mask_corners=False)
    if mask_lost == True:
        data_4D.mask[0, 0, 0, 0] = True
    if mask_fixed == True:
        data_4D.mask[-1, -1, -1, -1] = True
    return data_4D


def split_4D_to_5D_4(sfs, n4new, n5, model_update=True):
    """
    Four-to-five population split for the spectrum,
    n4 >= n4new+n5.

    sfs : 4D spectrum

    n4new : sample size for resulting pop 4

    n5 : sample size for resulting pop 5

    model_update : update model if set to True. Since we wrap this function to
        split other populations, we update the ModelPlot model outside of this
        function instead, and set it to False in that case.

    Returns a new 5D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0, 0, 0, 0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1, -1, -1, -1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None and model_update is True:
        model.split(3, (3, 4))

    data_4D = copy.copy(sfs)
    if len(data_4D.shape) != 4:
        raise ValueError("SFS is not 4D")
    n1 = data_4D.shape[0] - 1
    n2 = data_4D.shape[1] - 1
    n3 = data_4D.shape[2] - 1
    n4 = data_4D.shape[3] - 1
    if n4 < n4new + n5:
        raise ValueError("Requested sample sizes are too large")
    # if the sample size before split is too large, we project
    if n4 > n4new + n5:
        data_4D = data_4D.project([n1, n2, n3, n4new + n5])
    data_4D.unmask_all()

    # then we compute the joint fs resulting from the split
    data_5D = np.zeros((n1 + 1, n2 + 1, n3 + 1, n4new + 1, n5 + 1))
    for i in range(n4new + 1):
        for j in range(n5 + 1):
            log_entry_weight = (
                _log_comb(n4new, i) + _log_comb(n5, j) - _log_comb(n4new + n5, i + j)
            )
            data_5D[:, :, :, i, j] = data_4D.data[:, :, :, i + j] * np.exp(
                log_entry_weight
            )

    data_5D = Spectrum_mod.Spectrum(data_5D, mask_corners=False)
    if mask_lost == True:
        data_5D.mask[0, 0, 0, 0, 0] = True
    if mask_fixed == True:
        data_5D.mask[-1, -1, -1, -1, -1] = True
    return data_5D


def split_4D_to_5D_3(sfs, n3new, n5):
    """
    Four-to-five population split for the spectrum,
    n3 >= n3new+n4.

    sfs : 4D spectrum

    n3new : sample size for resulting pop 3

    n5 : sample size for resulting pop 4

    Returns a new 5D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0, 0, 0, 0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1, -1, -1, -1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(2, (2, 4))

    data_4D = copy.copy(sfs)
    if len(data_4D.shape) != 4:
        raise ValueError("SFS is not 4D")
    n1 = data_4D.shape[0] - 1
    n2 = data_4D.shape[1] - 1
    n3 = data_4D.shape[2] - 1
    n4 = data_4D.shape[3] - 1
    if n3 < n3new + n5:
        raise ValueError("Requested sample sizes are too large")
    # if the sample size before split is too large, we project
    if n3 > n3new + n5:
        data_4D = data_4D.project([n1, n2, n3new + n5, n4])
    data_4D.unmask_all()

    # then we compute the joint fs resulting from the split
    data_5D = np.zeros((n1 + 1, n2 + 1, n3new + 1, n4 + 1, n5 + 1))
    for i in range(n3new + 1):
        for j in range(n5 + 1):
            log_entry_weight = (
                _log_comb(n3new, i) + _log_comb(n5, j) - _log_comb(n3new + n5, i + j)
            )
            data_5D[:, :, i, :, j] = data_4D.data[:, :, i + j, :] * np.exp(
                log_entry_weight
            )

    data_5D = Spectrum_mod.Spectrum(data_5D, mask_corners=False)
    if mask_lost == True:
        data_5D.mask[0, 0, 0, 0, 0] = True
    if mask_fixed == True:
        data_5D.mask[-1, -1, -1, -1, -1] = True
    return data_5D


"""
Additional 3D and 4D splits, computed by swapping axes and applying existing
split functions above.
"""


def split_3D_to_4D_1(sfs, n1new, n4):
    """
    Uses split_3D_to_4D_3,
    swap 1st and 3rd population, split, and then swap back
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(0, (0, 3))

    fs = sfs.swapaxes(0, 2)
    fs = split_3D_to_4D_3(fs, n1new, n4, model_update=False)
    fs = fs.swapaxes(0, 2)
    return fs


def split_3D_to_4D_2(sfs, n2new, n4):
    """
    Uses split_3D_to_4D_3,
    swap 2nd and 3rd population, split, and then swap back
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(1, (1, 3))

    fs = sfs.swapaxes(1, 2)
    fs = split_3D_to_4D_3(fs, n2new, n4, model_update=False)
    fs = fs.swapaxes(1, 2)
    return fs


def split_4D_to_5D_1(sfs, n1new, n5):
    """
    Uses split_3D_to_4D_3,
    swap 1st and 3rd population, split, and then swap back
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(0, (0, 4))

    fs = sfs.swapaxes(0, 3)
    fs = split_4D_to_5D_4(fs, n1new, n5, model_update=False)
    fs = fs.swapaxes(0, 3)
    return fs


def split_4D_to_5D_2(sfs, n2new, n5):
    """
    Uses split_3D_to_4D_3,
    swap 1st and 3rd population, split, and then swap back
    """
    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.split(1, (1, 4))

    fs = sfs.swapaxes(1, 3)
    fs = split_4D_to_5D_4(fs, n2new, n5, model_update=False)
    fs = fs.swapaxes(1, 3)
    return fs


def split_by_index(sfs, idx, n0, n1):
    """
    Depending on the size of the SFS and the index that is split, we call
    one of the split functions. This acts as a wrapper for the various split
    functions, and is called by the Spectrum class function
    `fs.split(idx, n0, n1)`.
    """
    Npop = sfs.Npop
    if idx < 0 or idx + 1 > Npop:
        raise ValueError(f"Cannot split index {idx} in SFS of size {sfs.Npop}")

    if Npop == 1:
        sfs_split = split_1D_to_2D(sfs, n0, n1)
    elif Npop == 2:
        if idx == 0:
            sfs_split = split_2D_to_3D_1(sfs, n0, n1)
        else:
            sfs_split = split_2D_to_3D_2(sfs, n0, n1)
    elif Npop == 3:
        if idx == 0:
            sfs_split = split_3D_to_4D_1(sfs, n0, n1)
        elif idx == 1:
            sfs_split = split_3D_to_4D_2(sfs, n0, n1)
        else:
            sfs_split = split_3D_to_4D_3(sfs, n0, n1)
    elif Npop == 4:
        if idx == 0:
            sfs_split = split_4D_to_5D_1(sfs, n0, n1)
        elif idx == 1:
            sfs_split = split_4D_to_5D_2(sfs, n0, n1)
        elif idx == 2:
            sfs_split = split_4D_to_5D_3(sfs, n0, n1)
        else:
            sfs_split = split_4D_to_5D_4(sfs, n0, n1)
    return sfs_split


# merge two populations into one population
def merge_2D_to_1D(sfs):
    """
    Two-to-one populations fusion

    sfs : 2D spectrum

    Returns a new 1D spectrum
    """
    # Check if corners masked - if they are, keep split corners masked
    # If they are unmasked, keep split spectrum corners unmasked
    if sfs.mask[0, 0] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[-1, -1] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        model.merge((0, 1), 0)

    data_2D = copy.copy(sfs)
    assert len(data_2D.shape) == 2
    data_2D.unmask_all()
    dim1, dim2 = data_2D.shape
    data = np.zeros(dim1 + dim2 - 1)
    for k in range(dim1):
        for l in range(dim2):
            data[k + l] += data_2D[k, l]

    data = Spectrum_mod.Spectrum(data, mask_corners=False)
    if mask_lost == True:
        data.mask[0] = True
    if mask_fixed == True:
        data.mask[-1] = True
    return data


#  Methods for admixture


def __drop_last_slice__(sfs, dimension):
    # drop last slice along dimension in sfs

    ns = sfs.shape
    dim = len(ns)
    if dimension < 0:
        dimension = dim + dimension
    slicing = (
        (slice(None),) * dimension
        + (slice(None, -1),)
        + (slice(None),) * (dim - 1 - dimension)
    )
    return slicing


def __drop_first_slice__(sfs, dimension):
    # drop first slice along dimension in sfs
    ns = sfs.shape
    dim = len(ns)
    if dimension < 0:
        dimension = dim + dimension
    slicing = (
        (slice(None),) * dimension
        + (slice(1, None),)
        + (slice(None),) * (dim - 1 - dimension)
    )
    return slicing


def __migrate_1__(sfs, source_population_index, target_population_index):
    """
    Takes SFS, picks one individual from population source_population_index
    and migrates it to population target_population_index. If sfs has dimension
    (m,n), the new sfs will have dimension (m-1,n+1).
    """

    ns = sfs.shape
    new_ns = list(ns)
    M = ns[source_population_index] - 1
    N = ns[target_population_index] - 1

    new_ns[source_population_index] -= 1
    new_ns[target_population_index] += 1
    new_sfs = Spectrum_mod.Spectrum(np.zeros(new_ns), pop_ids=sfs.pop_ids)

    # We first suppose that we pick a reference allele.

    # since we picked a reference allele, there can be no contribution from the
    # sfs[:,:,ns[source_population_index],:,:], which would have all alts

    new_sfs[__drop_last_slice__(new_sfs, target_population_index)] = (
        sfs[__drop_last_slice__(sfs, source_population_index)].swapaxes(
            source_population_index, -1
        )
        * (1 - np.arange(M) * 1.0 / M)
    ).swapaxes(source_population_index, -1)

    new_sfs[__drop_first_slice__(new_sfs, target_population_index)] += (
        sfs[__drop_first_slice__(sfs, source_population_index)].swapaxes(
            source_population_index, -1
        )
        * (np.arange(1, M + 1) * 1.0 / M)
    ).swapaxes(source_population_index, -1)

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

    # It's unclear to me (APR) what is going on
    # as far as which nnls function gets called
    if sp.__version__ < "1.12":
        try:
            x, rnorm, mode = _nnls.nnls(A, m, n, b, w, zz, index, -1)
        except:
            x, rnorm, mode = _nnls.nnls(A, m, n, b, w, zz, index)
    else:
        x, rnorm = _nnls.nnls(A, b)
        mode = 1

    if mode != 1:
        warnings.warn(
            "Too many iterations in nnls. This usually occurs under "
            "strong migration models, and is often fixed by increasing "
            "the sample size prior to a large pulse migration event."
        )

    return x, rnorm


def __Gamma__(n_draws, n_lineages):
    """
    The gamma matrix element i,j gives the probability that a sequential sample
    of i lineages with replacement gives j distinct lineages
    """
    # the first row is the probability that a sample of 0 lineages gives j distinct
    # lineages: it is always 0 distinct lineages
    current = np.zeros(n_lineages + 1)
    current[0] = 1
    # then we compute the other rows through dynamic programming, adding one other sample
    # at a time: if we have a sample of size n, there are two possibilities for sample of
    # size n+1: either we draw an existing allele, or we draw a new allele.

    # then we compute the other rows through dynamic programming, adding one other sample
    # at a time: if had drawn n replacements, there are two possibilities for the n+1th
    # replacement: either we draw an existing allele, or we draw a new allele.

    list_arrays = [current]
    transition_matrix = np.diag(
        [i * 1.0 / n_lineages for i in range(n_lineages + 1)]
    ) + np.diag([1 - i * 1.0 / n_lineages for i in range(n_lineages)], k=-1)

    for i in range(n_draws):
        list_arrays.append(np.dot(transition_matrix, list_arrays[-1]))
    return np.array(list_arrays)


# Admixture of population 1 and 2 into a new population [-1], using the exact dp approach


def admix_into_new(sfs, dimension1, dimension2, n_lineages, m1, new_dimension=None):
    """
    Creates n_lineages in a new dimension to the SFS by drawing each from
    populations indexed by dimension1 (with probability m1) and dimension2
    (with probability 1-m1).

    The resulting frequency spectrum has
    - (dimension1 - n_lineages) lineages in dimension 1
    - (dimension2 - n_lineages) lineages in dimension 2
    - (n_lineages) lineages in new dimension

    By default, the new population is assigned the last dimension. We may wish
    to place the new population between the two admixed groups, and can specify
    the new dimension to place the admixed population. Note that this doesn't
    matter for integration, but to more naturally plot the model using
    ModelPlot.

    :param sfs: the input frequency spectrum object
    :param int dimension1: integer index of first source population
    :param int dimension2: integer index of second source population
    :param int n_lineages: number of lineages in the new admixed population
    :param float m1: proportion of lineages drawn from source
    :param int new_dimension: by default places the new population in the last
        dimension, but can be placed elsewhere in the order of population IDs by
        specifying an index
    """
    # Check if corners are masked - if they are, keep corners masked after event
    # If they are unmasked, keep spectrum corners unmasked after event
    if sfs.mask[tuple([0 for d in sfs.shape])] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[tuple([-1 for d in sfs.shape])] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # remove pop_ids so that we don't get ValueErrors
    pop_ids = sfs.pop_ids
    sfs.pop_ids = None

    dimensions = sfs.shape
    new_dimensions = list(dimensions) + [1]
    M = dimensions[dimension1] - 1
    N = dimensions[dimension2] - 1
    new_sfs = sfs.reshape(new_dimensions)

    if n_lineages > min(M, N):
        raise ValueError(
            "not enough lineages to produce %d, M=%d,N=%d"
            % (
                n_lineages,
                M,
                N,
            )
        )
    project_dimensions = [
        n - 1 for n in new_dimensions
    ]  # projection use number of lineages

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        if new_dimension == None:
            model.admix_new((dimension1, dimension2), len(new_dimensions) - 1, m1)
        else:
            model.admix_new((dimension1, dimension2), new_dimension, m1)

    for _i in range(n_lineages):
        project_dimensions[-1] += 1
        project_dimensions[dimension1] -= 1
        project_dimensions[dimension2] -= 1
        # print "pd", project_dimensions
        # print (m1 * migrate_1(new_sfs, dimension1,-1)).shape
        # print ((1-m1) * migrate_1(new_sfs, dimension2,-1)).shape
        new_sfs = Spectrum_mod.Spectrum.project(
            m1 * __migrate_1__(new_sfs, dimension1, -1), project_dimensions
        ) + Spectrum_mod.Spectrum.project(
            (1 - m1) * __migrate_1__(new_sfs, dimension2, -1), project_dimensions
        )
    new_sfs = np.squeeze(new_sfs)  # Remove empty dimensions

    if new_dimension is not None:
        # we need to place the new (last) dimension at given dimension, swapping population indices
        new_sfs = np.moveaxis(new_sfs, -1, new_dimension)

    # Set masking in corners based on mask_lost and mask_fixed
    if mask_lost is False:
        new_sfs.mask[tuple([0 for d in new_sfs.shape])] = False
    else:
        new_sfs.mask[tuple([0 for d in new_sfs.shape])] = True
    if mask_fixed is False:
        new_sfs.mask[tuple([-1 for d in new_sfs.shape])] = False
    else:
        new_sfs.mask[tuple([-1 for d in new_sfs.shape])] = True
    sfs.pop_ids = pop_ids

    return new_sfs


# Approximate admixture model


def admix_inplace(sfs, source_population_index, target_population_index, keep_1, m1):
    """

    Admixes from source_population to target_population in place, sending
    migrants one by one, and normalizing so that in the end we have
    approximately the correct distribution of replaced lineages.

    :param sfs: the input spectrum object
    :param int source_population_index: integer index of source population
    :param int target_population_index: integer index of target population
    :param int keep_1: number of lineages from the source population that we
        want to keep tracking after admixture.
    :param float m1: proportion of offspring in target population drawn from
        parents in source population. Note that the number of tracked lineages
        in the sample that have migrated is a random variable!
    """
    # Check if corners are masked - if they are, keep corners masked after event
    # If they are unmasked, keep spectrum corners unmasked after event
    if sfs.mask[tuple([0 for d in sfs.shape])] == True:
        mask_lost = True
    else:
        mask_lost = False
    if sfs.mask[tuple([-1 for d in sfs.shape])] == True:
        mask_fixed = True
    else:
        mask_fixed = False

    # remove pop_ids so that we don't get ValueErrors
    pop_ids = sfs.pop_ids
    sfs.pop_ids = None

    dimensions = sfs.shape
    M = (
        dimensions[source_population_index] - 1
    )  # number of haploid samples is size of sfs - 1
    N = dimensions[target_population_index] - 1

    target_M = keep_1
    target_N = N

    target_dimensions = list(np.array(dimensions[:]) - 1)
    target_dimensions[source_population_index] = target_M
    target_dimensions[target_population_index] = target_N

    if keep_1 > M:
        raise ValueError(
            (
                "Cannot keep more lineages than we started with, keep_1=%d,\
    M=%d"
                % (keep_1, M)
            )
        )

    # Update ModelPlot if necessary
    model = ModelPlot._get_model()
    if model is not None:
        if new_dimension == None:
            model.admix_inpace(source_population_index, target_population_index, m1)
        else:
            model.admix_inpace(source_population_index, target_population_index, m1)

    ############################
    # We first compute the sequence of SFSs we would obtain by migrating individuals
    # sequentially. This will give us a range of distributions, which we will use to
    # compute the correct distribution below.

    max_replacements = M - keep_1

    current_sfs = sfs[:]

    list_sfs = [sfs.project(target_dimensions)]  # Remember the SFSs we computed
    list_replacements = [0]  # The number of replacements in the corresponding sfs

    for num_replacements in range(1, max_replacements + 1):
        # The shape of the sfs is (n1+1, n2+1,...). We want to extract
        # sample sizes (n1,n2,...)
        project_dimensions = [shape_elem - 1 for shape_elem in current_sfs.shape]

        project_dimensions[target_population_index] -= 1  #  since there is a migrant,
        # only n2-1 lineages from 2 survive

        # first remove one sample from population 2, then migrate one from pop 1 to pop 2
        current_sfs = __migrate_1__(
            Spectrum_mod.Spectrum.project(current_sfs, project_dimensions),
            source_population_index,
            target_population_index,
        )
        current_sfs.unmask_all()

        keeper_function = True  #  Eventually we may want to only keep a subset --
        #  but don't want to optimize too early.
        if keeper_function:
            list_sfs.append(current_sfs.project(target_dimensions))
            list_replacements.append(num_replacements)

    ##################
    # Now that we have computed the list of SFSs with sequential migrations, we want to
    # use them to compute the correct frequency spectrum

    gamma = __Gamma__(max_replacements, N)  # the conversion matrix giving us the num of
    # replacements after 0,1,...,max_replacements
    # replacements
    target = np.array([stats.binom(N, m1).pmf(i) for i in range(N + 1)])  # binomial is
    # the standard, but we could use any distribution!

    weights = __nnls_mod__(gamma.transpose(), target)  # find a positive definite set of
    # parameters that imitates the target
    if weights[1] > 0.001:
        warnings.warn(
            f"In binomial distribution approximation is {weights[1]:2.3f}, "
            "consider including more lineages. If more lineages don't resolve the "
            "situation, consider using the exact admixture model (admix_into_new)."
        )
    # Following could be optimized by making it a dot product
    new_sfs = 0
    for i in range(len(weights[0])):
        new_sfs += list_sfs[i] * weights[0][i]

    # Set masking in corners based on mask_lost and mask_fixed
    if mask_lost is False:
        new_sfs.mask[tuple([0 for d in new_sfs.shape])] = False
    else:
        new_sfs.mask[tuple([0 for d in new_sfs.shape])] = True
    if mask_fixed is False:
        new_sfs.mask[tuple([-1 for d in new_sfs.shape])] = False
    else:
        new_sfs.mask[tuple([-1 for d in new_sfs.shape])] = True
    sfs.pop_ids = pop_ids

    return new_sfs
