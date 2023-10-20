import numpy as np
import scipy as sp
from scipy.sparse import linalg
import copy
import moments.Spectrum_mod
from . import Numerics
import Jackknife as jk
import LinearSystem_1D as ls1
import LinearSystem_2D as ls2
from . import Reversible

# ------------------------------------------------------------------------------
# Functions for the computation of the Phi-moments for multidimensional models:
# we integrate the ode system on the Phi_n(i) to compute their evolution
# we write it (and solve it) as an approximated linear system:
#       Phi_n' = Bn(N) + (1/(4N)Dn + S1n + S2n)Phi_n
# where :
#       N is the total population size
#       Bn(N) is the mutation source term
#       1/(4N)Dn is the drift effect matrix
#       S1n is the selection matrix for h = 0.5
#       S2n is the effect of h != 0.5
# ------------------------------------------------------------------------------


# -----------------------------------
# functions to compute the matrices-
# -----------------------------------
# Mutations
def _calcB(dims, u):
    # u is a list of mutation rates in each population
    # allows for different mutation rates in different pops
    B = np.zeros(dims)
    for k in range(len(dims)):
        ind = np.zeros(len(dims), dtype="int")
        ind[k] = int(1)
        tp = tuple(ind)
        B[tp] = (dims[k] - 1) * u[k]
    return B


# Drift
def _calcD(dims):
    """
    dims : List containing the pop sizes

    Returns a list of drift matrices for each pair of pops
    """
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(
                [
                    ls2.calcD1(np.array([dims[i], dims[j]])),
                    ls2.calcD2(np.array([dims[i], dims[j]])),
                ]
            )
    return res


def _buildD(vd, dims, N):
    """
    Builds the effective drift matrices by multiplying by the 1/4N coeff

    vd : List containing the drift matrices

    dims : List containing the pop sizes

    N : List containing the effective pop sizes for each pop

    Returns a list of effective drift matrices for each pair of pops
    """
    if len(dims) == 1:
        return [1.0 / 4 / N[0] * vd[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(1.0 / (4 * N[i]) * vd[ctr][0] + 1.0 / (4 * N[j]) * vd[ctr][1])
            ctr += 1
    return res


# Selection 1
def _calcS(dims, ljk):
    """
    dims : List containing the pop sizes

    ljk : List containing the 1 jump jackknife matrices for each pair of pop

    Returns a list of selection matrices for each pair of pops
    """
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(
                [
                    ls2.calcS_1(np.array([dims[i], dims[j]]), ljk[i]),
                    ls2.calcS_2(np.array([dims[i], dims[j]]), ljk[j]),
                ]
            )
    return res


def _buildS(vs, dims, s, h):
    """
    Builds the effective selection matrices by multiplying by the correct coeff

    vs : List containing the selection matrices

    dims : List containing the pop sizes

    s : List containing the selection coefficients

    h : List containing the dominance coefficients

    Returns a list of effective selection matrices for each pair of pops
    """
    if len(dims) == 1:
        return [vs[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(s[i] * h[i] * vs[ctr][0] + s[j] * h[j] * vs[ctr][1])
            ctr += 1
    return res


# Selection 2
def _calcS2(dims, ljk):
    """
    dims : List containing the pop sizes

    ljk : List containing the 2 jumps jackknife matrices for each pair of pop

    Returns a list of selection matrices for each pair of pops
    """
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(
                [
                    ls2.calcS2_1(np.array([dims[i], dims[j]]), ljk[i]),
                    ls2.calcS2_2(np.array([dims[i], dims[j]]), ljk[j]),
                ]
            )
    return res


# Over/Under-dominance
def _calcUnderdominance(dims, ljk):
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(
                [
                    ls2.calcUnderdominance_1(np.array([dims[i], dims[j]]), ljk[i]),
                    ls2.calcUnderdominance_2(np.array([dims[i], dims[j]]), ljk[j]),
                ]
            )
    return res


def _buildS2(vs, dims, s, h):
    """
    Builds the effective selection matrices (part due to dominance)
    by multiplying by the correct coeff

    vs : List containing the selection matrices

    dims : List containing the pop sizes

    s : List containing the selection coefficients

    h : List containing the dominance coefficients

    Returns a list of effective selection matrices for each pair of pops
    """
    if len(dims) == 1:
        return [vs[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(
                s[i] * (1 - 2.0 * h[i]) * vs[ctr][0]
                + s[j] * (1 - 2.0 * h[j]) * vs[ctr][1]
            )
            ctr += 1
    return res


def _buildS3(vs, dims, o, h):
    """
    Builds the effective selection matrices (for symmetric overdominance)
    by multiplying by the correct coeff

    vs : List containing the selection matrices

    dims : List containing the pop sizes

    o : List containing the selection coefficients

    h : List containing the dominance coefficients

    Returns a list of effective selection matrices for each pair of pops
    """
    if len(dims) == 1:
        return [vs[0][0]]
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(h[i] * o[i] * vs[ctr][0] + h[j] * o[j] * vs[ctr][1])
            ctr += 1
    return res


# Migrations
def _calcM(dims, ljk):
    """
    dims : List containing the pop sizes

    ljk : List containing the 1 jump jackknife matrices for each pair of pop

    Returns a list of migration matrices for each pair of pops
    """
    res = []
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(
                [
                    ls2.calcM_1(np.array([dims[i], dims[j]]), ljk[j]),
                    ls2.calcM_2(np.array([dims[i], dims[j]]), ljk[i]),
                ]
            )
    return res


def _buildM(vm, dims, m):
    """
    Builds the effective migration matrices by multiplying by the migration coeff

    vm : List containing the migration matrices

    dims : List containing the pop sizes

    m : matrix containing the migration coefficients

    Returns a list of effective migration matrices for each pair of pops
    """
    res = []
    ctr = 0
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            res.append(m[i, j] * vm[ctr][0] + m[j, i] * vm[ctr][1])
            ctr += 1
    return res


# ----------------------------------
# updates for the time integration-
# ----------------------------------
# we solve a system like PX = QY
# step 1 functions correspond to the QY computation
# and step 2 to the resolution of PX = Y'


# 2D
# step 1
def _ud1_2pop_1(sfs, Q, dims):
    sfs = Q[0].dot(sfs.reshape(dims[0] * dims[1])).reshape(dims)
    return sfs


# step 2
def _ud2_2pop_1(sfs, slv, dims):
    sfs = (slv[0](sfs.reshape(dims[0] * dims[1]))).reshape(dims)
    return sfs


# for 3D, 4D and 5D cases, each couple of directions are coded separately to simplify the permutations...
# ------------------------------
# 3D


# step 1
def _ud1_3pop_1(sfs, Q, dims):
    for i in range(int(dims[2])):
        sfs[:, :, i] = (
            Q[0].dot(sfs[:, :, i].reshape(dims[0] * dims[1])).reshape(dims[0], dims[1])
        )
    return sfs


def _ud1_3pop_2(sfs, Q, dims):
    for i in range(int(dims[1])):
        sfs[:, i, :] = (
            Q[1].dot(sfs[:, i, :].reshape(dims[0] * dims[2])).reshape(dims[0], dims[2])
        )
    return sfs


def _ud1_3pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        sfs[i, :, :] = (
            Q[2].dot(sfs[i, :, :].reshape(dims[1] * dims[2])).reshape(dims[1], dims[2])
        )
    return sfs


# step 2
def _ud2_3pop_1(sfs, slv, dims):
    for i in range(int(dims[2])):
        sfs[:, :, i] = slv[0](sfs[:, :, i].reshape(dims[0] * dims[1])).reshape(
            dims[0], dims[1]
        )
    return sfs


def _ud2_3pop_2(sfs, slv, dims):
    for i in range(int(dims[1])):
        sfs[:, i, :] = slv[1](sfs[:, i, :].reshape(dims[0] * dims[2])).reshape(
            dims[0], dims[2]
        )
    return sfs


def _ud2_3pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        sfs[i, :, :] = slv[2](sfs[i, :, :].reshape(dims[1] * dims[2])).reshape(
            dims[1], dims[2]
        )
    return sfs


# ------------------------------
# 4D


# step 1
def _ud1_4pop_1(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:, :, i, j] = (
                Q[0]
                .dot(sfs[:, :, i, j].reshape(dims[0] * dims[1]))
                .reshape(dims[0], dims[1])
            )
    return sfs


def _ud1_4pop_2(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:, i, :, j] = (
                Q[1]
                .dot(sfs[:, i, :, j].reshape(dims[0] * dims[2]))
                .reshape(dims[0], dims[2])
            )
    return sfs


def _ud1_4pop_3(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:, i, j, :] = (
                Q[2]
                .dot(sfs[:, i, j, :].reshape(dims[0] * dims[3]))
                .reshape(dims[0], dims[3])
            )
    return sfs


def _ud1_4pop_4(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i, :, :, j] = (
                Q[3]
                .dot(sfs[i, :, :, j].reshape(dims[1] * dims[2]))
                .reshape(dims[1], dims[2])
            )
    return sfs


def _ud1_4pop_5(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i, :, j, :] = (
                Q[4]
                .dot(sfs[i, :, j, :].reshape(dims[1] * dims[3]))
                .reshape(dims[1], dims[3])
            )
    return sfs


def _ud1_4pop_6(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i, j, :, :] = (
                Q[5]
                .dot(sfs[i, j, :, :].reshape(dims[2] * dims[3]))
                .reshape(dims[2], dims[3])
            )
    return sfs


# step 2


def _ud2_4pop_1(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            sfs[:, :, i, j] = slv[0](
                sfs[:, :, i, j].reshape(dims[0] * dims[1])
            ).reshape(dims[0], dims[1])
    return sfs


def _ud2_4pop_2(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            sfs[:, i, :, j] = slv[1](
                sfs[:, i, :, j].reshape(dims[0] * dims[2])
            ).reshape(dims[0], dims[2])
    return sfs


def _ud2_4pop_3(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            sfs[:, i, j, :] = slv[2](
                sfs[:, i, j, :].reshape(dims[0] * dims[3])
            ).reshape(dims[0], dims[3])
    return sfs


def _ud2_4pop_4(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            sfs[i, :, :, j] = slv[3](
                sfs[i, :, :, j].reshape(dims[1] * dims[2])
            ).reshape(dims[1], dims[2])
    return sfs


def _ud2_4pop_5(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            sfs[i, :, j, :] = slv[4](
                sfs[i, :, j, :].reshape(dims[1] * dims[3])
            ).reshape(dims[1], dims[3])
    return sfs


def _ud2_4pop_6(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            sfs[i, j, :, :] = slv[5](
                sfs[i, j, :, :].reshape(dims[2] * dims[3])
            ).reshape(dims[2], dims[3])
    return sfs


# ------------------------------
# 5D


# step 1
def _ud1_5pop_1(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :, :] = (
                    Q[9]
                    .dot(sfs[i, j, k, :, :].reshape(dims[3] * dims[4]))
                    .reshape(dims[3], dims[4])
                )
    return sfs


def _ud1_5pop_2(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k, :] = (
                    Q[8]
                    .dot(sfs[i, j, :, k, :].reshape(dims[2] * dims[4]))
                    .reshape(dims[2], dims[4])
                )
    return sfs


def _ud1_5pop_3(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[4])):
                sfs[i, j, :, :, k] = (
                    Q[7]
                    .dot(sfs[i, j, :, :, k].reshape(dims[2] * dims[3]))
                    .reshape(dims[2], dims[3])
                )
    return sfs


def _ud1_5pop_4(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k, :] = (
                    Q[6]
                    .dot(sfs[i, :, j, k, :].reshape(dims[1] * dims[4]))
                    .reshape(dims[1], dims[4])
                )
    return sfs


def _ud1_5pop_5(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[i, :, j, :, k] = (
                    Q[5]
                    .dot(sfs[i, :, j, :, k].reshape(dims[1] * dims[3]))
                    .reshape(dims[1], dims[3])
                )
    return sfs


def _ud1_5pop_6(sfs, Q, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[i, :, :, j, k] = (
                    Q[4]
                    .dot(sfs[i, :, :, j, k].reshape(dims[1] * dims[2]))
                    .reshape(dims[1], dims[2])
                )
    return sfs


def _ud1_5pop_7(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k, :] = (
                    Q[3]
                    .dot(sfs[:, i, j, k, :].reshape(dims[0] * dims[4]))
                    .reshape(dims[0], dims[4])
                )
    return sfs


def _ud1_5pop_8(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[:, i, j, :, k] = (
                    Q[2]
                    .dot(sfs[:, i, j, :, k].reshape(dims[0] * dims[3]))
                    .reshape(dims[0], dims[3])
                )
    return sfs


def _ud1_5pop_9(sfs, Q, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, i, :, j, k] = (
                    Q[1]
                    .dot(sfs[:, i, :, j, k].reshape(dims[0] * dims[2]))
                    .reshape(dims[0], dims[2])
                )
    return sfs


def _ud1_5pop_10(sfs, Q, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, :, i, j, k] = (
                    Q[0]
                    .dot(sfs[:, :, i, j, k].reshape(dims[0] * dims[1]))
                    .reshape(dims[0], dims[1])
                )
    return sfs


# step 2
def _ud2_5pop_1(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[2])):
                sfs[i, j, k, :, :] = slv[9](
                    sfs[i, j, k, :, :].reshape(dims[3] * dims[4])
                ).reshape(dims[3], dims[4])
    return sfs


def _ud2_5pop_2(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[3])):
                sfs[i, j, :, k, :] = slv[8](
                    sfs[i, j, :, k, :].reshape(dims[2] * dims[4])
                ).reshape(dims[2], dims[4])
    return sfs


def _ud2_5pop_3(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[1])):
            for k in range(int(dims[4])):
                sfs[i, j, :, :, k] = slv[7](
                    sfs[i, j, :, :, k].reshape(dims[2] * dims[3])
                ).reshape(dims[2], dims[3])
    return sfs


def _ud2_5pop_4(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[i, :, j, k, :] = slv[6](
                    sfs[i, :, j, k, :].reshape(dims[1] * dims[4])
                ).reshape(dims[1], dims[4])
    return sfs


def _ud2_5pop_5(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[i, :, j, :, k] = slv[5](
                    sfs[i, :, j, :, k].reshape(dims[1] * dims[3])
                ).reshape(dims[1], dims[3])
    return sfs


def _ud2_5pop_6(sfs, slv, dims):
    for i in range(int(dims[0])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[i, :, :, j, k] = slv[4](
                    sfs[i, :, :, j, k].reshape(dims[1] * dims[2])
                ).reshape(dims[1], dims[2])
    return sfs


def _ud2_5pop_7(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[3])):
                sfs[:, i, j, k, :] = slv[3](
                    sfs[:, i, j, k, :].reshape(dims[0] * dims[4])
                ).reshape(dims[0], dims[4])
    return sfs


def _ud2_5pop_8(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[2])):
            for k in range(int(dims[4])):
                sfs[:, i, j, :, k] = slv[2](
                    sfs[:, i, j, :, k].reshape(dims[0] * dims[3])
                ).reshape(dims[0], dims[3])
    return sfs


def _ud2_5pop_9(sfs, slv, dims):
    for i in range(int(dims[1])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, i, :, j, k] = slv[1](
                    sfs[:, i, :, j, k].reshape(dims[0] * dims[2])
                ).reshape(dims[0], dims[2])
    return sfs


def _ud2_5pop_10(sfs, slv, dims):
    for i in range(int(dims[2])):
        for j in range(int(dims[3])):
            for k in range(int(dims[4])):
                sfs[:, :, i, j, k] = slv[0](
                    sfs[:, :, i, j, k].reshape(dims[0] * dims[1])
                ).reshape(dims[0], dims[1])
    return sfs


# update nD with permutations
def _update_step1(sfs, Q, dims, order):
    assert len(sfs.shape) == len(dims)
    assert len(Q) == len(dims) * (len(dims) - 1) / 2
    for i in order:
        sfs = eval("_ud1_" + str(len(dims)) + "pop_" + str(i + 1) + "(sfs, Q, dims)")
    return sfs


def _update_step2(sfs, slv, dims, order):
    assert len(sfs.shape) == len(dims)
    assert len(slv) == len(dims) * (len(dims) - 1) / 2
    for i in order:
        sfs = eval("_ud2_" + str(len(dims)) + "pop_" + str(i + 1) + "(sfs, slv, dims)")
    return sfs


def _permute(tab):
    res = tab[1:]
    res.append(tab[0])
    return res


# timestep computation


def compute_dt_bis(N, T, drift, selmig, dims):
    if callable(N):
        Nmin = N(0)  # np.amin(N(0), N(T))
    else:
        Nmin = N
    nbp = int(len(dims) * (len(dims) - 1) / 2)
    D = _buildD(drift, dims, Nmin)
    Mat = [(D[i] + selmig[i]).todense() for i in range(nbp)]
    ev = [np.linalg.eigvals(Mat[i]) for i in range(nbp)]
    return 0


def _compute_dt_1pop(N, m, s, h, timescale_factor=0.15):
    maxVM = max(
        0.25 / N,
        max(m),
        abs(s)
        * 2
        * max(
            np.abs(h + (1 - 2 * h) * 0.5) * 0.5 * (1 - 0.5),
            np.abs(h + (1 - 2 * h) * 0.25) * 0.25 * (1 - 0.25),
        ),
    )
    if maxVM > 0:
        dt = timescale_factor / maxVM
    else:
        dt = np.inf
    if dt == 0:
        raise ValueError(
            "Timestep is zero. Values passed in are N=%f, m=%s,"
            "s=%f, h=%f." % (N, str(m), s, h)
        )
    return dt


def compute_dt(N, m=None, s=None, h=None, timescale_factor=0.1):
    # def compute_dt(N, m=None, s=None, h=None, timescale_factor=0.05):
    if m is None:
        m = np.zeros([len(N), len(N)])
    if s is None:
        s = np.zeros(len(N))
    if h is None:
        h = 0.5 * np.ones(len(N))
    timesteps = [
        _compute_dt_1pop(N[i], m[i, :], s[i], h[i], timescale_factor)
        for i in range(len(N))
    ]
    return min(timesteps)


# ------------------------------
# Manipulate parameters


def _make_array_sel_dom(num_pops, gamma, h, overdominance=None):
    """
    For any input of gamma and h, return parameters in array form.
    Inputs could be None, scalar numbers, list of numberss,
    or numpy arrays of numbers.
    """
    # neutral case if the parameters are not provided
    if gamma is None:
        gamma = np.zeros(num_pops)
    if h is None:
        h = 0.5 * np.ones(num_pops)
    if overdominance is None:
        overdominance = np.zeros(num_pops)

    # we convert s and h into numpy arrays
    if callable(gamma):
        if hasattr(gamma(0), "__len__"):
            s = lambda t: np.array(gamma(t))
        else:
            s = lambda t: np.array([gamma(t)] * num_pops)
    elif hasattr(gamma, "__len__"):
        s = np.array(gamma)
    else:
        s = np.array([gamma] * num_pops)
    if callable(h):
        h_func = copy.copy(h)
        if hasattr(h(0), "__len__"):
            h = lambda t: np.array(h_func(t))
        else:
            h = lambda t: np.array([h_func(t)] * num_pops)
    else:
        if hasattr(h, "__len__"):
            h = np.array(h)
        else:
            h = np.array([h] * num_pops)
    if callable(overdominance):
        o_func = copy.copy(overdominance)
        if hasattr(overdominance(0), "__len__"):
            overdominance = lambda t: np.array(overdominance_func(t))
        else:
            overdominance = lambda t: np.array([overdominance_func(t)] * num_pops)
    else:
        if hasattr(overdominance, "__len__"):
            overdominance = np.array(overdominance)
        else:
            overdominance = np.array([overdominance] * num_pops)
    return s, h, overdominance


def _set_up_mutation_rates(num_pops, theta, theta_fd, theta_bd, finite_genome):
    """
    Mutation rates are scaled and returned as appropriately formatted arrays.
    If theta is single value, mutation rate is the same in each population.
    For the infinite sites model, the backward mutation rate `v` is set to None.
    """
    if not finite_genome:
        if hasattr(theta, "__len__"):
            u = np.array(theta) / 4.0
        else:
            u = np.array([theta / 4.0] * num_pops)
        v = None
    else:
        if hasattr(theta_fd, "__len__"):
            u = np.array(theta_fd) / 4.0
        else:
            u = np.array([theta_fd / 4.0] * num_pops)
        if hasattr(theta_bd, "__len__"):
            v = np.array(theta_bd) / 4.0
        else:
            v = np.array([theta_bd / 4.0] * num_pops)
    return u, v


def _apply_frozen_pops(frozen, finite_genome, u, v, Npop, s=None, mm=None):
    """
    If any populations are frozen, we set their population extremely large,
    selection to zero, and mutations to zero in those pops. If migration is
    also given, we set migration in and out of frozen populations to zero.
    """
    frozen = np.array(frozen)
    if np.any(frozen):
        frozen_pops = np.where(np.array(frozen) == True)[0]
        if s is not None:
            # fix selection
            if callable(s):
                s_func = copy.copy(s)
                s = lambda t: s_func(t) * (1 - frozen)
            else:
                for pop_num in frozen_pops:
                    s[pop_num] = 0.0
        # fix population sizes
        if callable(Npop):
            nu_func = copy.copy(Npop)
            Npop = lambda t: list(np.array(nu_func(t)) * (1 - frozen) + 1e40 * frozen)
        else:
            for pop_num in frozen_pops:
                Npop[pop_num] = 1e40
        # fix mutation to zero in frozen populations
        u *= 1 - frozen
        if finite_genome:
            v *= 1 - frozen
        if mm is not None:
            # fix migration to zero to and from frozen populations
            def fix_migrations(mig_matrix):
                for pop_num in frozen_pops:
                    mig_matrix[:, pop_num] = 0.0
                    mig_matrix[pop_num, :] = 0.0
                return mig_matrix

            if callable(mm):
                mm_func = copy.copy(mm)
                mm = lambda t: fix_migrations(mm_func(t))
            else:
                mm = fix_migrations(mm)
    return u, v, Npop, s, mm


def integrate_nD(
    sfs0,
    Npop,
    tf,
    dt_fac=0.1,
    gamma=None,
    h=None,
    overdominance=None,
    m=None,
    theta=1.0,
    adapt_dt=False,
    finite_genome=False,
    theta_fd=None,
    theta_bd=None,
    frozen=[False],
):
    """
    Integrate the SFS data array, allowing for migration between populations.

    If we integrate under the finite genome model (i.e. with recurrent/reversible
    mutations), theta_fd and theta_bd can be provided. If they are not given, we
    assume equal forward and backward mutation rates, equal to theta.

    :param sfs0: SFS data for p populations, as a p-dimensional array.
    :param Npop: Relative population sizes (relative to some effective size Ne),
        provided as a vector N = [N1,...,Np]. This can be given as a function that
        returns such a vector, allowing for user-defined time-dependent size
        functions.
    :param tf: Total simulation time in genetic time units (2Ne generations).
    :param dt_fac: Sets an upper bound for the time steps used in integration,
        relative to `tf`. A smaller value provides a more accurate solution when
        migration or selection are nonzero, but is slower as it would require
        more iterations.
    :param gamma: Scaled selection coefficients, which can be either a number
        or a vector gamma = [gamma1,...,gammap], allowing for different selection
        coefficients in each population. This can also be given as function
        returning a number or such a vector, allowing for selection coefficients
        to change over time.
    :param h: Dominance coefficients (h=1/2 implies additive selection). Can be
        given as a number or a vector h = [h1,...,hp], or a function that returns
        a number or vector of coefficients, allowing for dominance coefficients
        to change over time.
    :param overdominance: Scaled selection coefficient that is applied only to
        heterozygotes, in a selection system with fitnesses 1:1+s:1. Underdominance
        can be modeled by passing a negative value. Not that this is a symmetric
        under/over-dominance model, in which homozygotes for either the ancestral
        or derived allele have equal fitness. `gamma`, `h`, and `overdominance`
        can be combined (additively) to implement non-symmetric selection
        scenarios.
    :param m: Matrix of migration rates as a 2D array, with size pxp. Entry
        m[i,j] is the migration rate from pop j to pop i, forward in time,
        normalized by 1/4Ne.
    :param theta: The population size-scaled mutation rate 4*Ne*u.
    :param finite_genome: If true, integrate under the finite genome (i.e.,
        reversible mutation) model.
    :param theta_fd: The forward scaled mutation rate 4*Ne*u. Only used with
        finite_genome as True.
    :param theta_bd: The backward scaled mutation rate 4*Ne*v. Only used with
        finite_genome as True.
    :param frozen: A list of length equal to the number of populations in the SFS,
        specifying which populations are frozen, such as ancient samples. A
        population indicated as frozen (by True) will have drift, selection,
        mutation, and migration to and from that population turned off.
    """
    sfs0 = np.array(sfs0)
    n = np.array(sfs0.shape) - 1
    num_pops = len(n)

    s, h, overdominance = _make_array_sel_dom(num_pops, gamma, h, overdominance)

    Tmax = tf * 2.0
    dt = Tmax * dt_fac

    # dimensions of the sfs
    dims = np.array(n + np.ones(num_pops), dtype=int)
    d = int(np.prod(dims))

    u, v = _set_up_mutation_rates(num_pops, theta, theta_fd, theta_bd, finite_genome)

    if callable(m):
        mm = lambda t: np.array(m(t)) / 2.0
    else:
        mm = np.array(m) / 2.0

    # if any populations are frozen, we set their population extremely large,
    # selection to zero, and mutations to zero in those pops
    frozen = np.array(frozen)
    if np.any(frozen):
        u, v, Npop, s, mm = _apply_frozen_pops(frozen, finite_genome, u, v, Npop, s, mm)

    # parameters of the equation
    if callable(Npop):
        N = np.array(Npop(0))
    else:
        N = np.array(Npop)

    if np.any(N <= 0):
        raise ValueError("All population sizes must be positive")

    Nold = N.copy()
    Neff = N

    if callable(mm):
        mig = mm(0)
    else:
        mig = mm
    mig_old = mig.copy()

    # number of "directions" for the splitting
    nbp = int(len(n) * (len(n) - 1) / 2)
    if len(n) == 1:
        nbp = 1
    # we compute the matrices we will need
    ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(len(dims))]
    ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(len(dims))]

    # drift
    vd = _calcD(dims)
    D = _buildD(vd, dims, N)

    # selection
    if callable(s):
        s_new = s(0)
    else:
        s_new = s
    s_old = s_new.copy()
    if callable(h):
        h_new = h(0)
    else:
        h_new = h
    h_old = h_new.copy()
    if callable(overdominance):
        o_new = overdominance(0)
    else:
        o_new = overdominance
    o_old = o_new.copy()

    # selection part 1
    vs = _calcS(dims, ljk)
    S1 = _buildS(vs, dims, s_new, h_new)

    # selection part 2
    vs2 = _calcS2(dims, ljk2)
    S2 = _buildS2(vs2, dims, s_new, h_new)

    vs3 = _calcUnderdominance(dims, ljk2)
    S3 = _buildS3(vs3, dims, o_new, h_new)

    # migration
    vm = _calcM(dims, ljk)
    Mi_bwd = _buildM(vm, dims, mig)

    # mutations
    if finite_genome == False:
        B = _calcB(dims, u)
    else:
        B = Reversible._calcB_FB(dims, u, v)

    # indexes for the permutation trick
    order = list(range(nbp))

    # time step splitting
    split_dt = 1.0
    if len(n) > 2:
        split_dt = 2.0 * len(n)

    # indicator of negative entries
    neg = False

    # time loop:
    t = 0.0
    sfs = sfs0

    while t < Tmax:
        dt_old = dt
        sfs_old = sfs
        if neg == False:
            dt = min(compute_dt(N, mig, s_new, h_new), Tmax * dt_fac)
        if t + dt > Tmax:
            dt = Tmax - t

        # we update the value of N if a function was provided as argument
        if callable(Npop):
            N = np.array(Npop((t + dt) / 2.0))
            if np.any(N <= 0):
                raise ValueError("All population sizes must be positive")

            Neff = Numerics.compute_N_effective(Npop, 0.5 * t, 0.5 * (t + dt))
            n_iter_max = 10
            n_iter = 0
            acceptable_change = 0.5
            while np.max(np.abs(N - Nold) / Nold) > acceptable_change:
                dt /= 2
                N = np.array(Npop((t + dt) / 2.0))
                Neff = Numerics.compute_N_effective(Npop, 0.5 * t, 0.5 * (t + dt))

                n_iter += 1
                if n_iter >= n_iter_max:
                    # failed to find timestep that kept population changes in check.
                    print(
                        "warning: large change size at time"
                        + " t = %2.2f in function integrate_nD" % (t,)
                    )

                    print("N_old, ", Nold, "N_new", N)
                    print("relative change", np.max(np.abs(N - Nold) / Nold))
                    print("This can cause issues in integration.")
                    print(
                        "consider reducing timestep factor dt_fac in integrate function"
                    )
                    print("currently %2.2f" % dt_fac)
                    break

        # update migration matrix if callable and check non-negative rates
        if callable(mm):
            mig = mm((t + dt / 2) / 2.0)
        if np.any(mig < 0):
            raise ValueError(f"Migration rate is below zero in matrix:\n{mig}")
        if callable(s):
            s_new = s((t + dt / 2) / 2.0)
        if callable(h):
            h_new = h((t + dt / 2) / 2.0)
        if callable(overdominance):
            o_new = overdominance((t + dt / 2) / 2.0)

        # we recompute the matrix only if any parameter has changed
        if (
            t == 0.0
            or (Nold != N).any()
            or dt != dt_old
            or neg == True
            or (mig != mig_old).any()
            or (s_new != s_old).any()
            or (h_new != h_old).any()
            or (o_new != o_old).any()
        ):
            D = _buildD(vd, dims, Neff)
            Mi = _buildM(vm, dims, mig)
            S1 = _buildS(vs, dims, s_new, h_new)
            S2 = _buildS2(vs2, dims, s_new, h_new)
            S3 = _buildS3(vs3, dims, o_new, h_new)
            # system inversion for backward scheme
            slv = [
                linalg.factorized(
                    sp.sparse.identity(S1[i].shape[0], dtype="float", format="csc")
                    - dt
                    / 2.0
                    / split_dt
                    * (
                        1.0 / (max(len(n), 2) - 1) * (D[i] + S1[i] + S2[i] + S3[i])
                        + Mi[i]
                    )
                )
                for i in range(nbp)
            ]
            # forward step
            Q = [
                sp.sparse.identity(S1[i].shape[0], dtype="float", format="csc")
                + dt
                / 2.0
                / split_dt
                * (1.0 / (max(len(n), 2) - 1) * (D[i] + S1[i] + S2[i] + S3[i]) + Mi[i])
                for i in range(nbp)
            ]

        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = Q[0].dot(sfs)
            if finite_genome == False:
                sfs = slv[0](sfs + dt * B)
            else:
                sfs = slv[0](sfs + (dt * B).dot(sfs))
        elif len(n) > 1:
            if finite_genome == False:
                for i in range(int(split_dt)):
                    sfs = _update_step1(sfs, Q, dims, order)
                    sfs += dt / split_dt * B
                    sfs = _update_step2(sfs, slv, dims, order)
                    order = _permute(order)
            else:
                for i in range(int(split_dt)):
                    sfs = _update_step1(sfs, Q, dims, order)
                    for j in range(len(n)):
                        sfs = sfs + (dt / split_dt * B[j]).dot(sfs.flatten()).reshape(
                            n + 1
                        )
                    sfs = _update_step2(sfs, slv, dims, order)
                    order = _permute(order)

        if (sfs < 0).any() and adapt_dt:
            neg = True
            if dt > min(compute_dt(N, mig, s, h), Tmax * dt_fac) / 8.0:
                dt *= 0.5
            sfs = sfs_old

        else:
            neg = False
            Nold = N
            t += dt
            mig_old = mig
            s_old = s_new
            h_old = h_new
            o_old = o_new

    if finite_genome == False:
        return moments.Spectrum_mod.Spectrum(sfs)
    else:
        return moments.Spectrum_mod.Spectrum(sfs, mask_corners=False)
