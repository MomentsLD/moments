import numpy as np
import scipy as sp
from scipy.sparse import linalg
from . import Reversible
import moments.Spectrum_mod
from . import Numerics
import Jackknife as jk
import LinearSystem_1D as ls1
import LinearSystem_2D as ls2
import Tridiag_solve as ts
from . import Integration
import copy

# ------------------------------------------------------------------------------
# Functions for the computation of the Phi-moments for multidimensional models
# without migrations:
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


# ----------------------------------
# updates for the time integration-
# ----------------------------------
# we solve a system like PX = QY
# step 1 functions correspond to the QY computation
# and step 2 to the resolution of PX = Y'


# 2D
# step 1
def _ud1_2pop_1(sfs, Q):
    sfs = Q[0].dot(sfs)
    return sfs


def _ud1_2pop_2(sfs, Q):
    sfs = Q[1].dot(sfs.transpose()).transpose()
    return sfs


# step 2
def _ud2_2pop_1(sfs, slv):
    for i in range(int(sfs.shape[1])):
        sfs[:, i] = slv[0](sfs[:, i])
    return sfs


def _ud2_2pop_2(sfs, slv):
    for i in range(int(sfs.shape[0])):
        sfs[i, :] = slv[1](sfs[i, :])
    return sfs


# for 3D, 4D and 5D cases, each couple of directions are coded separately to simplify the permutations...
# ------------------------------
# 3D
# step 1


def _ud1_3pop_1(sfs, Q):
    dims = sfs.shape
    dim2 = np.prod(dims[1::])
    sfs = Q[0].dot(sfs.reshape(dims[0], dim2)).reshape(dims)
    return sfs


def _ud1_3pop_2(sfs, Q):
    Q = [Q[1]]
    sfs = _ud1_3pop_1(np.transpose(sfs, (1, 0, 2)), Q)
    return np.transpose(sfs, (1, 0, 2))


def _ud1_3pop_3(sfs, Q):
    Q = [Q[2]]
    sfs = _ud1_3pop_1(np.transpose(sfs, (2, 1, 0)), Q)
    return np.transpose(sfs, (2, 1, 0))


# step 2
def _ud2_3pop_1(sfs, slv):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[1])):
            sfs[:, j, i] = slv[0](sfs[:, j, i])
    return sfs


def _ud2_3pop_2(sfs, slv):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[0])):
            sfs[j, :, i] = slv[1](sfs[j, :, i])
    return sfs


def _ud2_3pop_3(sfs, slv):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[0])):
            sfs[j, i, :] = slv[2](sfs[j, i, :])
    return sfs


# ------------------------------
# 4D
# step 1
def _ud1_4pop_1(sfs, Q):
    return _ud1_3pop_1(sfs, Q)


def _ud1_4pop_2(sfs, Q):
    Q = [Q[1]]
    sfs = _ud1_4pop_1(np.transpose(sfs, (1, 0, 2, 3)), Q)
    return np.transpose(sfs, (1, 0, 2, 3))


def _ud1_4pop_3(sfs, Q):
    Q = [Q[2]]
    sfs = _ud1_4pop_1(np.transpose(sfs, (2, 1, 0, 3)), Q)
    return np.transpose(sfs, (2, 1, 0, 3))


def _ud1_4pop_4(sfs, Q):
    Q = [Q[3]]
    sfs = _ud1_4pop_1(np.transpose(sfs, (3, 1, 2, 0)), Q)
    return np.transpose(sfs, (3, 1, 2, 0))


# step 2
def _ud2_4pop_1(sfs, slv):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[:, i, j, k] = slv[0](sfs[:, i, j, k])
    return sfs


def _ud2_4pop_2(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[i, :, j, k] = slv[1](sfs[i, :, j, k])
    return sfs


def _ud2_4pop_3(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                sfs[i, j, :, k] = slv[2](sfs[i, j, :, k])
    return sfs


def _ud2_4pop_4(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                sfs[i, j, k, :] = slv[3](sfs[i, j, k, :])
    return sfs


# ------------------------------
# 5D
# step 1


def _ud1_5pop_1(sfs, Q):
    return _ud1_3pop_1(sfs, Q)


def _ud1_5pop_2(sfs, Q):
    Q = [Q[1]]
    sfs = _ud1_5pop_1(np.transpose(sfs, (1, 0, 2, 3, 4)), Q)
    return np.transpose(sfs, (1, 0, 2, 3, 4))


def _ud1_5pop_3(sfs, Q):
    Q = [Q[2]]
    sfs = _ud1_5pop_1(np.transpose(sfs, (2, 1, 0, 3, 4)), Q)
    return np.transpose(sfs, (2, 1, 0, 3, 4))


def _ud1_5pop_4(sfs, Q):
    Q = [Q[3]]
    sfs = _ud1_5pop_1(np.transpose(sfs, (3, 1, 2, 0, 4)), Q)
    return np.transpose(sfs, (3, 1, 2, 0, 4))


def _ud1_5pop_5(sfs, Q):
    Q = [Q[4]]
    sfs = _ud1_5pop_1(np.transpose(sfs, (4, 1, 2, 3, 0)), Q)
    return np.transpose(sfs, (4, 1, 2, 3, 0))


# step 2
def _ud2_5pop_1(sfs, slv):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[:, i, j, k, l] = slv[0](sfs[:, i, j, k, l])
    return sfs


def _ud2_5pop_2(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, :, j, k, l] = slv[1](sfs[i, :, j, k, l])
    return sfs


def _ud2_5pop_3(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, :, k, l] = slv[2](sfs[i, j, :, k, l])
    return sfs


def _ud2_5pop_4(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, k, :, l] = slv[3](sfs[i, j, k, :, l])
    return sfs


def _ud2_5pop_5(sfs, slv):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[3])):
                    sfs[i, j, k, l, :] = slv[4](sfs[i, j, k, l, :])
    return sfs


# neutral case step 2 (tridiag solver)
# 2D
def _udn2_2pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        sfs[:, i] = ts.solve(A[0], Di[0], C[0], sfs[:, i])
    return sfs


def _udn2_2pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        sfs[i, :] = ts.solve(A[1], Di[1], C[1], sfs[i, :])
    return sfs


# 3D
def _udn2_3pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[1])):
            sfs[:, j, i] = ts.solve(A[0], Di[0], C[0], sfs[:, j, i])
    return sfs


def _udn2_3pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[2])):
        for j in range(int(sfs.shape[0])):
            sfs[j, :, i] = ts.solve(A[1], Di[1], C[1], sfs[j, :, i])
    return sfs


def _udn2_3pop_3(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[0])):
            sfs[j, i, :] = ts.solve(A[2], Di[2], C[2], sfs[j, i, :])
    return sfs


# 4D
def _udn2_4pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[:, i, j, k] = ts.solve(A[0], Di[0], C[0], sfs[:, i, j, k])
    return sfs


def _udn2_4pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                sfs[i, :, j, k] = ts.solve(A[1], Di[1], C[1], sfs[i, :, j, k])
    return sfs


def _udn2_4pop_3(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                sfs[i, j, :, k] = ts.solve(A[2], Di[2], C[2], sfs[i, j, :, k])
    return sfs


def _udn2_4pop_4(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                sfs[i, j, k, :] = ts.solve(A[3], Di[3], C[3], sfs[i, j, k, :])
    return sfs


# 5D
def _udn2_5pop_1(sfs, A, Di, C):
    for i in range(int(sfs.shape[1])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[:, i, j, k, l] = ts.solve(A[0], Di[0], C[0], sfs[:, i, j, k, l])
    return sfs


def _udn2_5pop_2(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[2])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, :, j, k, l] = ts.solve(A[1], Di[1], C[1], sfs[i, :, j, k, l])
    return sfs


def _udn2_5pop_3(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[3])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, :, k, l] = ts.solve(A[2], Di[2], C[2], sfs[i, j, :, k, l])
    return sfs


def _udn2_5pop_4(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[4])):
                    sfs[i, j, k, :, l] = ts.solve(A[3], Di[3], C[3], sfs[i, j, k, :, l])
    return sfs


def _udn2_5pop_5(sfs, A, Di, C):
    for i in range(int(sfs.shape[0])):
        for j in range(int(sfs.shape[1])):
            for k in range(int(sfs.shape[2])):
                for l in range(int(sfs.shape[3])):
                    sfs[i, j, k, l, :] = ts.solve(A[4], Di[4], C[4], sfs[i, j, k, l, :])
    return sfs


# sfs update
def _update_step1(sfs, Q):
    assert len(Q) == len(sfs.shape)
    for i in range(len(sfs.shape)):
        sfs = eval("_ud1_" + str(len(sfs.shape)) + "pop_" + str(i + 1) + "(sfs, Q)")
    return sfs


def _update_step2(sfs, slv):
    assert len(slv) == len(sfs.shape)
    for i in range(len(sfs.shape)):
        sfs = eval("_ud2_" + str(len(sfs.shape)) + "pop_" + str(i + 1) + "(sfs, slv)")
    return sfs


def _update_step2_neutral(sfs, A, Di, C):
    assert len(A) == len(sfs.shape)
    for i in range(len(sfs.shape)):
        sfs = eval(
            "_udn2_" + str(len(sfs.shape)) + "pop_" + str(i + 1) + "(sfs, A, Di, C)"
        )
    return sfs


def integrate_nomig(
    sfs0,
    Npop,
    tf,
    dt_fac=0.1,
    gamma=None,
    h=None,
    overdominance=None,
    theta=1.0,
    adapt_tstep=False,
    finite_genome=False,
    theta_fd=None,
    theta_bd=None,
    frozen=[False],
):
    """
    Integrate the SFS data array, without migration between populations.

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

    s, h, overdominance = Integration._make_array_sel_dom(
        num_pops, gamma, h, overdominance
    )

    Tmax = tf * 2.0
    dt = Tmax * dt_fac

    # dimensions of the sfs
    dims = np.array(n + np.ones(num_pops), dtype=int)
    d = int(np.prod(dims))

    u, v = Integration._set_up_mutation_rates(
        num_pops, theta, theta_fd, theta_bd, finite_genome
    )

    frozen = np.array(frozen)
    if np.any(frozen):
        u, v, Npop, s, _ = Integration._apply_frozen_pops(
            frozen, finite_genome, u, v, Npop, s
        )

    # parameters of the equation
    if callable(Npop):
        N = np.array(Npop(0))
    else:
        N = np.array(Npop)

    if np.any(N <= 0):
        raise ValueError("All population sizes must be positive")

    Nold = N.copy()
    # effective pop size for the integration
    Neff = N

    # we compute the matrices we will need
    ljk = [jk.calcJK13(int(dims[i] - 1)) for i in range(num_pops)]
    ljk2 = [jk.calcJK23(int(dims[i] - 1)) for i in range(num_pops)]

    # drift
    vd = [ls1.calcD(np.array(dims[i])) for i in range(num_pops)]
    D = [1.0 / 4 / N[i] * vd[i] for i in range(num_pops)]

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
    vs = [ls1.calcS(dims[i], ljk[i]) for i in range(num_pops)]
    S1 = [s_new[i] * h_new[i] * vs[i] for i in range(num_pops)]

    # selection part 2
    vs2 = [ls1.calcS2(dims[i], ljk2[i]) for i in range(num_pops)]
    S2 = [s_new[i] * (1 - 2.0 * h_new[i]) * vs2[i] for i in range(num_pops)]

    # overdominance term
    vs3 = [ls1.calcUnderdominance(dims[i], ljk2[i]) for i in range(num_pops)]
    S3 = [h_new[i] * o_new[i] * vs3[i] for i in range(num_pops)]

    # mutations
    if finite_genome == False:
        B = _calcB(dims, u)
    else:
        B = Reversible._calcB_FB(dims, u, v)

    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        dt_old = dt
        # dt = compute_dt(sfs.shape, N, gamma, h, 0, Tmax * dt_fac)
        dt = min(Integration.compute_dt(N, s=s_new, h=h_new), Tmax * dt_fac)
        if t + dt > Tmax:
            dt = Tmax - t

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
                if np.any(N <= 0):
                    raise ValueError("All population sizes must be positive")

                Neff = Numerics.compute_N_effective(Npop, 0.5 * t, 0.5 * (t + dt))

                n_iter += 1
                if n_iter >= n_iter_max:
                    # failed to find timestep that kept population shanges in check.
                    print(
                        "warning: large change size at time"
                        + " t = %2.2f in function integrate_nomig" % (t,)
                    )

                    print("N_old, ", Nold, "N_new", N)
                    print("relative change", np.max(np.abs(N - Nold) / Nold))
                    break

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
            or (s_new != s_old).any()
            or (h_new != h_old).any()
            or (o_new != o_old).any()
        ):
            D = [1.0 / 4 / Neff[i] * vd[i] for i in range(num_pops)]
            S1 = [s_new[i] * h_new[i] * vs[i] for i in range(num_pops)]
            S2 = [s_new[i] * (1 - 2.0 * h_new[i]) * vs2[i] for i in range(num_pops)]
            S3 = [h_new[i] * o_new[i] * vs3[i] for i in range(num_pops)]
            # system inversion for backward scheme
            slv = [
                linalg.factorized(
                    sp.sparse.identity(S1[i].shape[0], dtype="float", format="csc")
                    - dt / 2.0 * (D[i] + S1[i] + S2[i] + S3[i])
                )
                for i in range(len(n))
            ]
            Q = [
                sp.sparse.identity(S1[i].shape[0], dtype="float", format="csc")
                + dt / 2.0 * (D[i] + S1[i] + S2[i] + S3[i])
                for i in range(len(n))
            ]

        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            sfs = Q[0].dot(sfs)
            if finite_genome == False:
                sfs = slv[0](sfs + dt * B)
            else:
                sfs = slv[0](sfs + (dt * B).dot(sfs))
        elif len(n) > 1:
            sfs = _update_step1(sfs, Q)
            if finite_genome == False:
                sfs = sfs + dt * B
            else:
                for i in range(len(n)):
                    sfs = sfs + (dt * B[i]).dot(sfs.flatten()).reshape(n + 1)
            sfs = _update_step2(sfs, slv)
        Nold = N
        t += dt
        s_old = s_new
        h_old = h_new
        o_old = o_new

    if finite_genome == False:
        return moments.Spectrum_mod.Spectrum(sfs)
    else:
        return moments.Spectrum_mod.Spectrum(sfs, mask_corners=False)


def integrate_neutral(
    sfs0,
    Npop,
    tf,
    dt_fac=0.1,
    theta=1.0,
    adapt_tstep=False,
    finite_genome=False,
    theta_fd=None,
    theta_bd=None,
    frozen=[False],
):
    """
    Integrate the SFS data array, without migration between populations and
    no selection.

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

    Tmax = tf * 2.0
    dt = Tmax * dt_fac
    # dimensions of the sfs
    dims = np.array(n + np.ones(num_pops), dtype=int)
    d = int(np.prod(dims))

    u, v = Integration._set_up_mutation_rates(
        num_pops, theta, theta_fd, theta_bd, finite_genome
    )

    # if any populations are frozen, we set their population extremely large,
    # and mutations to zero in those pops
    frozen = np.array(frozen)
    if np.any(frozen):
        u, v, Npop, _, _ = Integration._apply_frozen_pops(
            frozen, finite_genome, u, v, Npop
        )

    # parameters of the equation
    if callable(Npop):
        N = np.array(Npop(0))
    else:
        N = np.array(Npop)

    if np.any(N <= 0):
        raise ValueError("All population sizes must be positive")

    Nold = N.copy()
    Neff = N

    # drift
    vd = [ls1.calcD_dense(dims[i]) for i in range(len(n))]
    diags = [ts.mat_to_diag(x) for x in vd]
    D = [1.0 / 4 / N[i] * vd[i] for i in range(len(n))]

    # mutations
    if finite_genome == False:
        B = _calcB(dims, u)
    else:
        B = Reversible._calcB_FB(dims, u, v)

    # time loop:
    t = 0.0
    sfs = sfs0
    while t < Tmax:
        dt_old = dt
        # dt = compute_dt(sfs.shape, N, 0, 0, 0, Tmax * dt_fac)
        dt = min(Integration.compute_dt(N), Tmax * dt_fac)
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
                if np.any(N <= 0):
                    raise ValueError("All population sizes must be positive")

                Neff = Numerics.compute_N_effective(Npop, 0.5 * t, 0.5 * (t + dt))

                n_iter += 1
                if n_iter >= n_iter_max:
                    # failed to find timestep that kept population shanges in check.
                    print(
                        "warning: large change size at time"
                        + " t = %2.2f in function integrate_neutral" % (t,)
                    )

                    print("N_old, ", Nold, "N_new", N)
                    print("relative change", np.max(np.abs(N - Nold) / Nold))
                    break

        # we recompute the matrix only if N has changed...
        if (
            t == 0.0 or (Nold != N).any() or dt != dt_old
        ):  # SG not sure why dt_old is involved here.
            D = [1.0 / 4 / Neff[i] * vd[i] for i in range(len(n))]
            A = [-0.5 * dt / 4 / Neff[i] * diags[i][0] for i in range(len(n))]
            Di = [
                np.ones(dims[i]) - 0.5 * dt / 4 / Neff[i] * diags[i][1]
                for i in range(len(n))
            ]
            C = [-0.5 * dt / 4 / Neff[i] * diags[i][2] for i in range(len(n))]
            # system inversion for backward scheme
            for i in range(len(n)):
                ts.factor(A[i], Di[i], C[i])
            Q = [np.eye(dims[i]) + 0.5 * dt * D[i] for i in range(len(n))]

        # drift, selection and migration (depends on the dimension)
        if len(n) == 1:
            if finite_genome == False:
                sfs = ts.solve(A[0], Di[0], C[0], np.dot(Q[0], sfs) + dt * B)
            else:
                sfs = ts.solve(A[0], Di[0], C[0], np.dot(Q[0], sfs) + (dt * B).dot(sfs))
        else:
            sfs = _update_step1(sfs, Q)
            if finite_genome == False:
                sfs = sfs + dt * B
            else:
                for i in range(len(n)):
                    sfs = sfs + (dt * B[i]).dot(sfs.flatten()).reshape(n + 1)
            sfs = _update_step2_neutral(sfs, A, Di, C)
        Nold = N
        t += dt

    if finite_genome == False:
        return moments.Spectrum_mod.Spectrum(sfs)
    else:
        return moments.Spectrum_mod.Spectrum(sfs, mask_corners=False)

    return Spectrum_mod.Spectrum(sfs)
