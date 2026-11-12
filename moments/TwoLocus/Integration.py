import numpy as np
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import spsolve
import moments.TwoLocus.Jackknife
import moments.TwoLocus.Numerics


def _resolve_selection(gamma, sel_params, sel_params_general, finite_genome=False):
    """
    Check that at most one selection model is specified, and convert ``gamma`` to
    ``sel_params`` under the infinite sites model.

    Returns (gamma, sel_params, sel_params_general), with selection inputs that are
    all zero replaced by None (and gamma = 0).
    """
    if gamma is None:
        gamma = 0.0
    if sel_params is not None and np.all([s == 0 for s in sel_params]):
        sel_params = None
    if sel_params_general is not None and np.all([s == 0 for s in sel_params_general]):
        sel_params_general = None

    if finite_genome:
        if sel_params is not None or sel_params_general is not None:
            raise ValueError(
                "With finite_genome=True, only gamma (additive selection at the left "
                "locus) is supported, not sel_params or sel_params_general"
            )
        return gamma, None, None

    num_models = (gamma != 0) + (sel_params is not None)
    num_models += sel_params_general is not None
    if num_models > 1:
        raise ValueError("Specify only one of gamma, sel_params, or sel_params_general")
    if gamma != 0:
        # additive selection with the same coefficient at both loci, no epistasis
        sel_params = [2 * gamma, gamma, gamma]
        gamma = 0.0
    return gamma, sel_params, sel_params_general


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
    Integrate the two-locus frequency spectrum forward in time. Users will typically
    call :meth:`moments.TwoLocus.TLSpectrum.integrate` instead, which takes the same
    arguments and is documented there.

    :param F: The two-locus frequency spectrum, as an array of shape (n+1, n+1, n+1).
    :returns: The integrated spectrum, as a numpy array of the same shape.
    """
    if rho is None:
        rho = 0
    if rho < 0:
        raise ValueError("rho must be non-negative")

    if tf < 0:
        raise ValueError("integration time must be non-negative")
    elif tf == 0:
        return F

    gamma, sel_params, sel_params_general = _resolve_selection(
        gamma, sel_params, sel_params_general, finite_genome=finite_genome
    )

    n = len(F) - 1

    if callable(nu):
        N = nu(0)
    else:
        N = nu

    N_old = 1.0

    compute_jk1 = False
    compute_jk2 = False
    # selection operators that act on the jump-1 and jump-2 jackknife extrapolations
    S1 = None
    S2 = None

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
        if sel_params is not None:
            compute_jk1 = True
            S1 = moments.TwoLocus.Numerics.selection_two_locus(n, sel_params)
        if sel_params_general is not None:
            compute_jk2 = True
            S2 = moments.TwoLocus.Numerics.selection_general(n, sel_params_general)
    else:
        if clustered_mutations:
            raise ValueError("clustered mutations only allowed in ISM")
        if u is None or v is None:
            raise ValueError("if finite_genome is True, must specify u and v")
        if alternate_fg == True:
            M, M2 = moments.TwoLocus.Numerics.mutations_reversible_2(n, u, v)
        else:
            M = moments.TwoLocus.Numerics.mutations_reversible(n, u, v)
        D = moments.TwoLocus.Numerics.drift_reversible(n)
        if gamma != 0:
            S1 = gamma * moments.TwoLocus.Numerics.selection_reversible_additive(n)
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
            if S1 is not None:
                Ab += S1.dot(J1)
            if S2 is not None:
                Ab += S2.dot(J2)
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
    Compute the steady-state two-locus frequency spectrum under the infinite sites
    model. Users will typically call :func:`moments.TwoLocus.Demographics.equilibrium`
    instead, which documents the parameters and can cache the result.

    :param int n: The sample size.
    :param float rho: The population-size scaled recombination rate, 4*Ne*r.
    :param float theta: The population-size scaled mutation rate at each locus, 4*Ne*u.
    :param list sel_params: Additive selection parameters [gamma_AB, gamma_A, gamma_B].
    :param list sel_params_general: Selection parameters for the nine diploid
        genotypes.
    :param bool clustered_mutations: If True, new mutations arise as AB haplotypes
        at both loci at once instead of on the background of the other locus.
    :returns: The equilibrium spectrum, as a :class:`moments.TwoLocus.TLSpectrum`.
    """
    if rho is None:
        rho = 0.0
    if rho < 0:
        raise ValueError("recombination rate must be non-negative")
    if theta <= 0:
        raise ValueError("theta must be positive")

    _, sel_params, sel_params_general = _resolve_selection(
        0, sel_params, sel_params_general
    )

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

    if sel_params is not None:
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
