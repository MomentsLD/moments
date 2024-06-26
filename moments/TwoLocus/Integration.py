import numpy as np
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import spsolve
import moments.TwoLocus.Jackknife
import moments.TwoLocus.Numerics

import warnings

warnings.filterwarnings("ignore")


def integrate(
    F,
    nu,
    tf,
    rho=0.0,
    dt=0.01,
    theta=1.0,
    gamma=0.0,
    sel_params=None,
    sel_params_general=None,
    finite_genome=False,
    u=None,
    v=None,
    alternate_fg=False,
    clustered_mutations=False,
):
    """
    There are two selection options:
    1) set gamma and h, which is for selection at the left locus, linked to a neutral locus
    2) set sel_params, which are the (additive) selection coefficients for haplotypes AB, Ab, and aB
        so that sel_params = (sAB, sA, sB)
        If sAB = sA + sB, no epistatic interaction
    Finite genome (reversible mutation) options are only available to selection and
    dominance at the left locus.
    The default selection model is available with the ISM model.
    """
    if rho is None:
        rho = 0
    if rho < 0:
        raise ValueError("rho must be non-negative")

    if tf < 0:
        raise ValueError("must have positive integration time")
    elif tf == 0:
        return F

    n = len(F) - 1

    if callable(nu):
        N = nu(0)
    else:
        N = nu

    N_old = 1.0

    compute_jk1 = False
    compute_jk2 = False

    if rho != 0:
        compute_jk1 = True
        if finite_genome is False:
            R = moments.TwoLocus.Numerics.recombination(n, rho)
        else:
            R = moments.TwoLocus.Numerics.recombination_reversible(n, rho)

    if finite_genome is False:
        if clustered_mutations:
            M_0to1 = moments.TwoLocus.Numerics.mutations_mnm(n, theta=theta)
        else:
            M_0to1, M = moments.TwoLocus.Numerics.mutations(n, theta=theta)
        D = moments.TwoLocus.Numerics.drift(n)
        if sel_params is None and gamma != 0:
            sel_params = [2 * gamma, gamma, gamma]
        if sel_params is not None:
            compute_jk1 = True
            S = moments.TwoLocus.Numerics.selection_two_locus(n, sel_params)
        if sel_params_general is not None:
            compute_jk2 = True
            S = moments.TwoLocus.Numerics.selection_general(n, sel_params_general)
    else:
        if clustered_mutations:
            raise ValueError("clustered mutations only allowed in ISM")
        if sel_params is not None:
            raise ValueError("if finite_genome is True, cannot use sel_params")
        if u is None or v is None:
            raise ValueError("if finite_genome is True, must specify u and v")
        if alternate_fg == True:
            M, M2 = moments.TwoLocus.Numerics.mutations_reversible_2(n, u, v)
        else:
            M = moments.TwoLocus.Numerics.mutations_reversible(n, u, v)
        D = moments.TwoLocus.Numerics.drift_reversible(n)
        if gamma != 0:
            S = gamma * moments.TwoLocus.Numerics.selection_reversible_additive(n)
            compute_jk1 = True

    if compute_jk1:
        J1 = moments.TwoLocus.Jackknife.calc_jk(n, 1)
    if compute_jk2:
        J2 = moments.TwoLocus.Jackknife.calc_jk(n, 2)

    Phi = moments.TwoLocus.Numerics.array_to_Phi(F)

    t_elapsed = 0

    while t_elapsed < tf:
        dt_old = dt
        if t_elapsed + dt > tf:
            dt = tf - t_elapsed

        if callable(nu):
            N = nu(t_elapsed + dt / 2.0)

        if t_elapsed == 0 or N_old != N or dt != dt_old:
            # recompute solver
            Ab = D / (2.0 * N)
            if not clustered_mutations:
                Ab += M
            if rho != 0:
                Ab += R.dot(J1)
            if (
                sel_params is not None and np.any([s != 0 for s in sel_params])
            ) or gamma > 0:
                Ab += S.dot(J1)
            elif sel_params_general is not None:
                Ab += S.dot(J2)
            Ab1 = identity(Ab.shape[0], format="csc") + dt / 2.0 * Ab
            slv = factorized(identity(Ab.shape[0], format="csc") - dt / 2.0 * Ab)

        if finite_genome is False:
            Phi = slv(Ab1.dot(Phi) + dt * M_0to1)
        else:
            Phi = slv(Ab1.dot(Phi))
            if alternate_fg == True:
                Phi = Phi + M2.dot(Phi)

        if np.any(Phi < -1):
            raise ValueError(
                f"Spectrum has large negative values, min = {Phi.min()}. "
                "Selection may be too strong or recombination too large. "
                "This may be fixed with a larger sample size, at the cost of runtime."
            )
        N_old = N
        t_elapsed += dt

    return moments.TwoLocus.Numerics.Phi_to_array(Phi, n)


# Use scipy sparse linear algebra solvers to compute the steady state solution
# for the ISM model, which requires removing the fixed bins.


def delete_rows_cols(A, b, indices):
    if A.shape[0] != A.shape[1]:
        raise ValueError("only apply to square matrix")
    if A.shape[0] != len(b):
        raise ValueError("dimension mismatch")
    indices = list(indices)
    mask = np.ones(A.shape[0], dtype=bool)
    mask[indices] = False
    remaining = np.arange(len(mask)).compress(mask)
    return A[mask][:, mask], b.compress(mask), remaining


def steady_state(
    n,
    rho=0.0,
    theta=1.0,
    sel_params=None,
    sel_params_general=None,
    clustered_mutations=False,
):
    """
    Compute the steady state distribution for the additive or general selection model
    and infinite sites.
    """
    if rho < 0:
        raise ValueError("recombination rate must be non-negative")
    if theta <= 0:
        raise ValueError("theta must be positive")

    if sel_params is not None and sel_params_general is not None:
        raise ValueError("pick one selection model or the other")

    if clustered_mutations:
        M_0to1 = moments.TwoLocus.Numerics.mutations_mnm(n, theta=theta)
    else:
        M_0to1, M = moments.TwoLocus.Numerics.mutations(n, theta=theta)
    D = moments.TwoLocus.Numerics.drift(n)

    Ab = D / 2.0
    if not clustered_mutations:
        Ab += M

    computed_jk1 = False
    if rho > 0:
        J1 = moments.TwoLocus.Jackknife.calc_jk(n, 1)
        R = moments.TwoLocus.Numerics.recombination(n, rho)
        computed_jk1 = True
        Ab += R.dot(J1)

    if sel_params is not None and np.any([s != 0 for s in sel_params]):
        S = moments.TwoLocus.Numerics.selection_two_locus(n, sel_params)
        if computed_jk1 is False:
            J1 = moments.TwoLocus.Jackknife.calc_jk(n, 1)
            computed_jk1 = True
        Ab += S.dot(J1)
    elif sel_params_general is not None:
        S = moments.TwoLocus.Numerics.selection_general(n, sel_params_general)
        J2 = moments.TwoLocus.Jackknife.calc_jk(n, 2)
        Ab += S.dot(J2)

    to_del = [moments.TwoLocus.Numerics.index_n(n, 0, 0, 0)]
    for i in range(0, n):
        j = n - i
        to_del.append(moments.TwoLocus.Numerics.index_n(n, i, j, 0))
        to_del.append(moments.TwoLocus.Numerics.index_n(n, i, 0, j))
    to_del.append(moments.TwoLocus.Numerics.index_n(n, n, 0, 0))

    A, b, indices = delete_rows_cols(Ab, M_0to1, to_del)
    sts = spsolve(A, -b)

    sts_full = np.zeros(Ab.shape[0])
    sts_full[indices] = sts
    F = moments.TwoLocus.Numerics.Phi_to_array(sts_full, n)
    return moments.TwoLocus.TLSpectrum(F)
