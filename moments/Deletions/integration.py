# Functions for initializing and integrating the deletion frequency
# spectrum forward in time.

from . import util

from scipy.sparse.linalg import factorized
from scipy.sparse import identity
from scipy.special import gammaln
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.special as scisp
from mpmath import hyp1f1

import copy
import pickle
import numpy as np
import os


def _valid_mutation_rates(theta):
    if hasattr(theta, "len"):
        if len(theta) != 2:
            raise ValueError("thetas must be float or list of length 2")
        if np.any([_ < 0 for _ in theta]):
            raise ValueError("mutation rates must be positive")
    else:
        if theta < 0:
            raise ValueError("mutation rate must be positive")
        else:
            theta = [theta, theta]
    return theta


def integrate_crank_nicolson(
    data, nu, T, dt=0.01, theta_snp=0.001, theta_del=0.001, sel_coeffs=None,
):
    """
    Integrate the frequency spectrum using the Crank-Nicolson scheme.

    We have a reversible mutation model, with SNP and SV mutation rates given by
    *per-base* :math:`theta`.

    :param data: The deletion spectrum object data.
    :type data: array-like
    :param nu:
    :type nu: float or function returning float
    :param T:
    :type T: float
    :param dt:
    :type dt: float, optional
    :param theta_snp: If a single value, we assume equal forward and backward mutation
        rate. Defaults to 0.001. If a list of length two is given, it specifies the
        forward and backward mutation rates, resp.
    :type theta_snp: float or list, optional
    :param theta_del: If a single value, we assume equal forward and backward mutation
        rate. Defaults to 0.001. If a list of length two is given, it specifies the
        forward and backward mutation rates, resp.
    :type theta_del: float or list, optional
    """
    theta_snp = _valid_mutation_rates(theta_snp)
    theta_del = _valid_mutation_rates(theta_del)

    if T < 0:
        raise ValueError("integration time must be positive")
    elif T == 0:
        return data
    else:
        data = copy.copy(data)

    if not callable(nu):
        if nu <= 0:
            raise ValueError("population size must be positive")
        N0 = N1 = nu

    N0_prev = 0
    N1_prev = 0

    n_float = util.get_n_from_length(len(data))
    n = int(np.rint(n_float))
    assert np.isclose(n_float, n)

    D = drift_matrix(n)
    U = mutation_matrix(n, theta_snp, theta_del)

    if sel_coeffs is not None:
        raise ValueError("need to implement this")
        S = selection_matrix(n, sel_coeffs)
        J = calcJK_2(n)

    t_elapsed = 0
    while t_elapsed < T:
        # at some point, might want to implement adaptive time steps.
        # for now, we don't

        if t_elapsed + dt > T:
            dt = T - t_elapsed

        if callable(nu):
            N0 = nu(t_elapsed)
            N1 = nu(t_elapsed + dt)

        if t_elapsed == 0 or N0_prev != N0 or N1_prev != N1 or dt != dt_prev:
            Ab0 = D / (2 * N0) + U
            Ab1 = D / (2 * N1) + U
            if sel_coeffs is not None:
                Ab0 += S.dot(J)
                Ab1 += S.dot(J)
            Ab_fwd = identity(Ab0.shape[0], format="csc") + dt / 2.0 * Ab0
            Ab_bwd = factorized(identity(Ab1.shape[0], format="csc") - dt / 2.0 * Ab1)

        data = Ab_bwd(Ab_fwd.dot(data))

        N0_prev = N0
        N1_prev = N1
        dt_prev = dt

        # check here for negative or nan values, for adaptive time stepping

        t_elapsed += dt

    return data


def equilibrium(
    n, nu=1, theta_snp=[0.001, 0.001], theta_del=[0.001, 0.001], sel_coeffs=None,
):
    """
    Returns the data array for the equilibrium spectrum for sample size n.
    """
    D = drift_matrix(n)
    U = mutation_matrix(n, theta_snp, theta_del)
    Ab = D / 2 / nu + U

    if sel_coeffs is not None:
        S = selection_matrix(n, sel_coeffs)
        J = calcJK_2(n)
        Ab += S.dot(J)

    tol = 1e-20
    dt = 0.1

    Ab_fwd = identity(Ab.shape[0], format="csc") + dt / 2.0 * Ab
    Ab_bwd = factorized(identity(Ab.shape[0], format="csc") - dt / 2.0 * Ab)

    data = np.ones((n + 1) * (n + 2) // 2)
    data /= len(data)

    data_last = np.ones((n + 1) * (n + 2) // 2)
    i = 0
    while np.sum((data - data_last) ** 2) > tol:
        i += 1
        data_last = data
        data = Ab_bwd(Ab_fwd.dot(data))
    print("required", i, "iterations")
    return data


####
# Integration transition matrices
####


def drift_matrix(n):
    D = np.zeros(((n + 1) * (n + 2) // 2,) * 2)
    for i in range(n + 1):
        for j in range(n + 1 - i):
            this_idx = util.get_idx(n, i, j)
            D[this_idx, this_idx] -= 2 * ((n - i - j) * i + (n - i - j) * j + i * j)
            if i < n and i + j + 1 <= n:
                D[util.get_idx(n, i + 1, j), this_idx] += (n - i - j) * i
            if i > 0:
                D[util.get_idx(n, i - 1, j), this_idx] += (n - i - j) * i
            if j < n and i + j + 1 <= n:
                D[util.get_idx(n, i, j + 1), this_idx] += (n - i - j) * j
            if j > 0:
                D[util.get_idx(n, i, j - 1), this_idx] += (n - i - j) * j
            if i < n and j > 0:
                D[util.get_idx(n, i + 1, j - 1), this_idx] += i * j
            if i > 0 and j < n:
                D[util.get_idx(n, i - 1, j + 1), this_idx] += i * j
    return csc_matrix(D)


def mutation_matrix(n, theta_snp, theta_del):
    """
    :param n: sample size
    :param theta_snp: forward and backward snp mutation rates (recurrent)
    :param theta_del: forward (deletion) and backward (insertion) rates, which
        only occur for the fixed categories.
    """
    U = np.zeros(((n + 1) * (n + 2) // 2,) * 2)

    # snp mutations
    for i in range(n):
        for j in range(n + 1 - i):
            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
                theta_snp[0] / 2 * (n - i - j)
            )
            if n - i - j > 0:
                U[util.get_idx(n, i, j + 1), util.get_idx(n, i, j)] += (
                    theta_snp[0] / 2 * (n - i - j)
                )
            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= theta_snp[1] / 2 * j
            if j > 0:
                U[util.get_idx(n, i, j - 1), util.get_idx(n, i, j)] += (
                    theta_snp[1] / 2 * j
                )

    # deletion mutations
    for j in range(n + 1):
        # del hits A
        if j > 0:
            U[util.get_idx(n, 0, j), util.get_idx(n, 0, j)] -= theta_del[0] / 2 * j
            U[util.get_idx(n, 1, j - 1), util.get_idx(n, 0, j)] += theta_del[0] / 2 * j
        # del hits a
        if n - j > 0:
            U[util.get_idx(n, 0, j), util.get_idx(n, 0, j)] -= (
                theta_del[0] / 2 * (n - j)
            )
            U[util.get_idx(n, 1, j), util.get_idx(n, 0, j)] += (
                theta_del[0] / 2 * (n - j)
            )

    # insertion mutations
    # from fixed deletions, inserts a mutation
    U[util.get_idx(n, n, 0), util.get_idx(n, n, 0)] -= theta_del[1] / 2 * n
    U[util.get_idx(n, n - 1, 0), util.get_idx(n, n, 0)] += theta_del[1] / 2 * n

    return csc_matrix(U)


#def mutation_matrix_recurrent(n, theta_snp, theta_del):
#    """
#    n: The sample size.
#    theta_snp: A list of length two, with theta_fwd and theta_bwd
#        for a->A and A->a mutations.
#    theta_del: A list of length two, with theta_fwd and theat_bwd
#        for deletions and insertions.
#    """
#    U = np.zeros(((n + 1) * (n + 2) // 2,) * 2)
#    for i in range(n + 1):
#        for j in range(n + 1 - i):
#            # mutation from a -> A, takes (i, j) -> (i + 1, j)
#            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
#                theta_snp[0] / 2 * (n - i - j)
#            )
#            if n - i - j > 0:
#                U[util.get_idx(n, i + 1, j), util.get_idx(n, i, j)] += (
#                    theta_snp[0] / 2 * (n - i - j)
#                )
#            # mutation from A -> a, takes (i, j) -> (i - 1, j)
#            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= theta_snp[1] / 2 * i
#            if i > 0:
#                U[util.get_idx(n, i - 1, j), util.get_idx(n, i, j)] += (
#                    theta_snp[1] / 2 * i
#                )
#            # deletion mutation, takes (i, j) to (i, j + 1) and (i - 1, j + 1)
#            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
#                theta_del[0] / 2 * i
#            )  # hits A
#            U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= (
#                theta_del[0] / 2 * (n - i - j)  # hits a
#            )
#            if i > 0:
#                U[util.get_idx(n, i - 1, j + 1), util.get_idx(n, i, j)] += (
#                    theta_del[0] / 2 * i
#                )
#            if (n - i - j) > 0:
#                U[util.get_idx(n, i, j + 1), util.get_idx(n, i, j)] += (
#                    theta_del[0] / 2 * (n - i - j)
#                )
#            # insertion mutation, takes (i, j) to (i, j - 1) and (i + 1, j - 1)
#            # insertions of derived and ancestral states are equally likely
#            if j > 0:
#                U[util.get_idx(n, i, j), util.get_idx(n, i, j)] -= theta_del[1] / 2 * j
#                U[util.get_idx(n, i, j - 1), util.get_idx(n, i, j)] += (
#                    theta_del[1] / 2 * j / 2
#                )
#                U[util.get_idx(n, i + 1, j - 1), util.get_idx(n, i, j)] += (
#                    theta_del[1] / 2 * j / 2
#                )
#    return csc_matrix(U)


def selection_matrix(n, sel_params):
    """
    To include selection with dominance, T_n depends on T_(n+2), which is
    estimated using the jackknife (calcJK_2). This means that the selection
    transition matrix has shape (size T_n) \times (size T_n+2).

    Selection operator goes like S.J.T

    :param n: The sample size.
    :param sel_params: List of selection coefficients for (Aa, AA, aX, AX, and XX)
        genotypes, where X denotes a deletion.
    """
    s_Aa, s_AA, s_aX, s_AX, s_XX = sel_params
    size_n = (n + 1) * (n + 2) // 2
    size_n_2 = (n + 3) * (n + 4) // 2
    S = np.zeros((size_n, size_n_2))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            fr_idx = util.get_idx(n, i, j)

            ## For each event, track outgoing density, and where it goes

            # (A a) A
            if n - i - j > 0:
                jk_idx = util.get_idx(n + 2, i, j + 2)
                to_idx = util.get_idx(n, i, j + 1)
                S[fr_idx, jk_idx] -= -s_Aa / 3 * (j + 2) * (j + 1) * (n - i - j)
                S[to_idx, jk_idx] += -s_Aa / 3 * (j + 2) * (j + 1) * (n - i - j)

            # (A a) X
            if n - i - j > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j + 1)
                to_idx = util.get_idx(n, i + 1, j)
                S[fr_idx, jk_idx] -= -s_Aa / 6 * (i + 1) * (j + 1) * (n - i - j)
                S[to_idx, jk_idx] += -s_Aa / 6 * (i + 1) * (j + 1) * (n - i - j)

            # (a A) a
            if j > 0:
                jk_idx = util.get_idx(n + 2, i, j)
                to_idx = util.get_idx(n, i, j - 1)
                S[fr_idx, jk_idx] -= -s_Aa / 3 * (j + 1) * (n - i - j + 1) * (n - i - j)
                S[to_idx, jk_idx] += -s_Aa / 3 * (j + 1) * (n - i - j + 1) * (n - i - j)

            # (a A) X
            if j > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j)
                to_idx = util.get_idx(n, i + 1, j - 1)
                S[fr_idx, jk_idx] -= -s_Aa / 6 * (i + 1) * j * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_Aa / 6 * (i + 1) * j * (n - i - j + 1)

            # (A A) a
            if j > 0:
                jk_idx = util.get_idx(n + 2, i, j + 1)
                to_idx = util.get_idx(n, i, j - 1)
                S[fr_idx, jk_idx] -= -s_AA / 3 * (j + 1) * j * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_AA / 3 * (j + 1) * j * (n - i - j + 1)

            # (A A) X
            if j > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j + 1)
                to_idx = util.get_idx(n, i + 1, j - 1)
                S[fr_idx, jk_idx] -= -s_AA / 3 * (i + 1) * (j + 1) * j
                S[to_idx, jk_idx] += -s_AA / 3 * (i + 1) * (j + 1) * j

            # (a X) a
            if i > 0:
                jk_idx = util.get_idx(n + 2, i, j)
                to_idx = util.get_idx(n, i - 1, j)
                S[fr_idx, jk_idx] -= -s_aX / 3 * i * (n - i - j + 2) * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_aX / 3 * i * (n - i - j + 2) * (n - i - j + 1)

            # (a X) A
            if i > 0:
                jk_idx = util.get_idx(n + 2, i, j + 1)
                to_idx = util.get_idx(n, i - 1, j + 1)
                S[fr_idx, jk_idx] -= -s_aX / 6 * i * (j + 1) * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_aX / 6 * i * (j + 1) * (n - i - j + 1)

            # (X a) A
            if n - i - j > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j + 1)
                to_idx = util.get_idx(n, i, j + 1)
                S[fr_idx, jk_idx] -= -s_aX / 6 * (i + 1) * (j + 1) * (n - i - j)
                S[to_idx, jk_idx] += -s_aX / 6 * (i + 1) * (j + 1) * (n - i - j)

            # (X a) X
            if n - i - j > 0:
                jk_idx = util.get_idx(n + 2, i + 2, j)
                to_idx = util.get_idx(n, i + 1, j)
                S[fr_idx, jk_idx] -= -s_aX / 3 * (i + 2) * (i + 1) * (n - i - j)
                S[to_idx, jk_idx] += -s_aX / 3 * (i + 2) * (i + 1) * (n - i - j)

            # (A X) a
            if i > 0:
                jk_idx = util.get_idx(n + 2, i, j + 1)
                to_idx = util.get_idx(n, i - 1, j)
                S[fr_idx, jk_idx] -= -s_AX / 6 * i * (j + 1) * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_AX / 6 * i * (j + 1) * (n - i - j + 1)

            # (A X) A
            if i > 0:
                jk_idx = util.get_idx(n + 2, i, j + 2)
                to_idx = util.get_idx(n, i - 1, j + 1)
                S[fr_idx, jk_idx] -= -s_AX / 3 * i * (j + 2) * (j + 1)
                S[to_idx, jk_idx] += -s_AX / 3 * i * (j + 2) * (j + 1)

            # (X A) a
            if j > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j)
                to_idx = util.get_idx(n, i, j - 1)
                S[fr_idx, jk_idx] -= -s_AX / 6 * (i + 1) * j * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_AX / 6 * (i + 1) * j * (n - i - j + 1)

            # (X A) X
            if j > 0:
                jk_idx = util.get_idx(n + 2, i + 2, j)
                to_idx = util.get_idx(n, i + 1, j - 1)
                S[fr_idx, jk_idx] -= -s_AX / 3 * (i + 2) * (i + 1) * j
                S[to_idx, jk_idx] += -s_AX / 3 * (i + 2) * (i + 1) * j

            # (X X) a
            if i > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j)
                to_idx = util.get_idx(n, i - 1, j)
                S[fr_idx, jk_idx] -= -s_XX / 3 * (i + 1) * i * (n - i - j + 1)
                S[to_idx, jk_idx] += -s_XX / 3 * (i + 1) * i * (n - i - j + 1)

            # (X X) A
            if i > 0:
                jk_idx = util.get_idx(n + 2, i + 1, j + 1)
                to_idx = util.get_idx(n, i - 1, j + 1)
                S[fr_idx, jk_idx] -= -s_XX / 3 * (i + 1) * i * (j + 1)
                S[to_idx, jk_idx] += -s_XX / 3 * (i + 1) * i * (j + 1)

    S /= (n + 2) * (n + 1)
    return csc_matrix(S)


####
# Jackknife functions, taken from moments.Triallele
####


# Cache jackknife matrices in ~/.moments/TwoLocus_cache by default
def set_cache_path(path="~/.deletions/jackknife_cache"):
    """
    Set directory in which jackknife matrices are cached, so they do not
    need to be recomputed each time.
    """
    global cache_path
    cache_path = os.path.expanduser(path)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)


cache_path = None
set_cache_path()


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
    # check if cached, if so just load it
    jackknife_fname = f"jk_{n}_2.mtx"
    if os.path.isfile(os.path.join(cache_path, jackknife_fname)):
        with open(os.path.join(cache_path, jackknife_fname), "rb") as fin:
            try:
                J = pickle.load(fin)
            except:
                J = pickle.load(fin, encoding="Latin1")
        return J

    # size of J is size of n+1 spectrum x size of n spectrum
    # J = np.zeros(((n+3)*(n+4)/2,(n+1)*(n+2)/2))
    row = []
    col = []
    data = []

    for i in range(1, n + 2):
        for j in range(1, n + 2 - i):
            ordered_set = closest_ij_2(i, j, n)
            alphas = compute_alphas_2(i, j, ordered_set, n)
            index2 = util.get_idx(n + 2, i, j)
            for pair, alpha in zip(ordered_set, alphas):
                index = util.get_idx(n, pair[0], pair[1])
                # J[index2,index] = alpha
                row.append(index2)
                col.append(index)
                data.append(alpha)

    # jackknife for the biallelic edges (i=0, j=1:n, and j=0, i=1:n)
    # first for j = 0
    j = 0
    for i in range(1, n + 2):
        this_ind = util.get_idx(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        # J[this_ind, util.get_idx(n,ii-1,j)] = alphas[0]
        # J[this_ind, util.get_idx(n,ii,j)] = alphas[1]
        # J[this_ind, util.get_idx(n,ii+1,j)] = alphas[2]
        row.append(this_ind)
        col.append(util.get_idx(n, ii - 1, 0))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(util.get_idx(n, ii, 0))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(util.get_idx(n, ii + 1, 0))
        data.append(alphas[2])

    i = 0
    for j in range(1, n + 2):
        this_ind = util.get_idx(n + 2, i, j)
        jj = find_iprime_1D(n, j)
        alphas = get_alphas_1D(jj, j, n)
        # J[this_ind, util.get_idx(n,i,jj-1)] = alphas[0]
        # J[this_ind, util.get_idx(n,i,jj)] = alphas[1]
        # J[this_ind, util.get_idx(n,i,jj+1)] = alphas[2]
        row.append(this_ind)
        col.append(util.get_idx(n, 0, jj - 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(util.get_idx(n, 0, jj))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(util.get_idx(n, 0, jj + 1))
        data.append(alphas[2])

    # jackknife along diagonal - 1D jk
    for i in range(1, n + 2):
        j = n + 2 - i
        this_ind = util.get_idx(n + 2, i, j)
        ii = find_iprime_1D(n, i)
        alphas = get_alphas_1D(ii, i, n)
        row.append(this_ind)
        col.append(util.get_idx(n, ii - 1, n - ii + 1))
        data.append(alphas[0])
        row.append(this_ind)
        col.append(util.get_idx(n, ii, n - ii))
        data.append(alphas[1])
        row.append(this_ind)
        col.append(util.get_idx(n, ii + 1, n - ii - 1))
        data.append(alphas[2])

    J = csr_matrix(
        (data, (row, col)),
        shape=(int((n + 3) * (n + 4) / 2), int((n + 1) * (n + 2) / 2)),
    )
    # cache J
    with open(os.path.join(cache_path, jackknife_fname), "wb+") as fout:
        pickle.dump(J, fout, pickle.HIGHEST_PROTOCOL)

    return J
