import numpy as np
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
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
    finite_genome=False,
    u=None,
    v=None,
    alternate_fg=False,
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
    if rho != 0:
        compute_jk1 = True
        if finite_genome is False:
            R = moments.TwoLocus.Numerics.recombination(n, rho)
        else:
            R = moments.TwoLocus.Numerics.recombination_reversible(n, rho)

    if finite_genome is False:
        M_0to1, M = moments.TwoLocus.Numerics.mutations(n, theta=theta)
        D = moments.TwoLocus.Numerics.drift(n)
        if sel_params is None and gamma != 0:
            sel_params = [2 * gamma, gamma, gamma]
        if sel_params is not None:
            compute_jk1 = True
            S = moments.TwoLocus.Numerics.selection_two_locus(n, sel_params)
    else:
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

    if compute_jk1:
        J1 = moments.TwoLocus.Jackknife.calc_jk(n, 1)

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
            Ab = M / 2.0 + D / (2.0 * N)
            if rho != 0:
                Ab += R.dot(J1)
            if (
                sel_params is not None and np.any([s != 0 for s in sel_params])
            ) or gamma > 0:
                Ab += S.dot(J1)
            Ab1 = identity(Ab.shape[0], format="csc") + dt / 2.0 * Ab
            slv = factorized(identity(Ab.shape[0], format="csc") - dt / 2.0 * Ab)

        if finite_genome is False:
            Phi = slv(Ab1.dot(Phi) + dt * M_0to1)
        else:
            Phi = slv(Ab1.dot(Phi))
            if alternate_fg == True:
                Phi = Phi + M2.dot(Phi)

        N_old = N
        t_elapsed += dt

    return moments.TwoLocus.Numerics.Phi_to_array(Phi, n)
