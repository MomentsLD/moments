import numpy as np, math
from scipy.special import gammaln
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from collections import defaultdict

# might want to save the projection caches - especially for larger sample sizes,
# faster to load than recalculate each time

index_cache = {}


def index_n(n, i, j, k):
    """
    For a spectrum of sample size n, take in an (n+1)^3 sized object, convert to
    correctly sized array Phi.
    Here, we'll try to operate on the whole spectrum, since recombination cannot be split.
    We need a dictionary that maps (i, j, k) to the correct index in the array.
    """
    try:
        return index_cache[n][(i, j, k)]
    except KeyError:
        indexes = {}
        indexes.setdefault(n, {})
        ll = 0
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    indexes[n][(ii, jj, kk)] = ll
                    ll += 1
        index_cache[n] = indexes[n]
        return index_cache[n][(i, j, k)]


def array_to_Phi(F):
    """
    The array F is the square masked array object, with lots of zeros for infeasible
    entries. Returns Phi, which is a 1D vector keeping variable entries.
    """
    n = len(F) - 1
    Phi = np.zeros(int((n + 1) * (n + 2) * (n + 3) / 6))
    for ii in range(n + 1):
        for jj in range(n + 1 - ii):
            for kk in range(n + 1 - ii - jj):
                Phi[index_n(n, ii, jj, kk)] = F[ii, jj, kk]
    return Phi


def Phi_to_array(Phi, n):
    F = np.zeros((n + 1, n + 1, n + 1))
    for ii in range(n + 1):
        for jj in range(n + 1 - ii):
            for kk in range(n + 1 - ii - jj):
                F[ii, jj, kk] = Phi[index_n(n, ii, jj, kk)]
    return F


# so now we need slice drift transition matrices for each slice (3*(n+1) of them)
# we track the background biallelic frequency spectra along the [0,:,0] and [0,0,:] axes,
# so we don't want density flowing back onto those points


def choose(n, i):
    return np.exp(gammaln(n + 1) - gammaln(n - i + 1) - gammaln(i + 1))


def drift(n):
    """
    The drift operator for sample size n.
    """
    Dsize = int((n + 1) * (n + 2) * (n + 3) / 6)
    row = []
    col = []
    data = []
    for ii in range(n + 1):
        for jj in range(n + 1 - ii):
            for kk in range(n + 1 - ii - jj):
                # skip if A/a or B/b fixed
                fA = ii + jj
                fa = n - (ii + jj)
                fB = ii + kk
                fb = n - (ii + kk)
                if fA == n or fB == n:
                    continue
                # single locus drift along B/b and A/a variable, with other fixed ancestral
                # fixed for b
                elif fA == 0:  # 0 <= kk <= n
                    if kk == 0 or kk == n:
                        continue
                    this_ind = index_n(n, ii, jj, kk)
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj, kk - 1))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (kk - 1) * (n - kk + 1) / n / (n - 1))
                    )
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj, kk))
                    data.append(2 * choose(n, 2) * (-2.0 * kk * (n - kk) / n / (n - 1)))
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj, kk + 1))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (kk + 1) * (n - kk - 1) / n / (n - 1))
                    )
                elif fB == 0:
                    if jj == 0 or jj == n:
                        continue
                    this_ind = index_n(n, ii, jj, kk)
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj - 1, kk))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (jj - 1) * (n - jj + 1) / n / (n - 1))
                    )
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj, kk))
                    data.append(2 * choose(n, 2) * (-2.0 * jj * (n - jj) / n / (n - 1)))
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj + 1, kk))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (jj + 1) * (n - jj - 1) / n / (n - 1))
                    )
                else:
                    this_ind = index_n(n, ii, jj, kk)
                    # incoming density
                    if ii > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii - 1, jj, kk))
                        data.append(
                            2
                            * choose(n, 2)
                            * (1.0 * (ii - 1) * (n - ii - jj - kk + 1) / n / (n - 1))
                        )

                    if n - ii - jj - kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii + 1, jj, kk))
                        data.append(
                            2
                            * choose(n, 2)
                            * (1.0 * (ii + 1) * (n - ii - jj - kk - 1) / n / (n - 1))
                        )

                    if ii > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii - 1, jj, kk + 1))
                        data.append(
                            2 * choose(n, 2) * (1.0 * (ii - 1) * (kk + 1) / n / (n - 1))
                        )

                    if kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii + 1, jj, kk - 1))
                        data.append(
                            2 * choose(n, 2) * (1.0 * (ii + 1) * (kk - 1) / n / (n - 1))
                        )

                    if ii > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii - 1, jj + 1, kk))
                        data.append(
                            2 * choose(n, 2) * (1.0 * (ii - 1) * (jj + 1) / n / (n - 1))
                        )

                    if jj > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii + 1, jj - 1, kk))
                        data.append(
                            2 * choose(n, 2) * (1.0 * (ii + 1) * (jj - 1) / n / (n - 1))
                        )

                    if jj > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii, jj - 1, kk))
                        data.append(
                            2
                            * choose(n, 2)
                            * (1.0 * (jj - 1) * (n - ii - jj - kk + 1) / n / (n - 1))
                        )

                    if n - ii - jj - kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii, jj + 1, kk))
                        data.append(
                            2
                            * choose(n, 2)
                            * (1.0 * (jj + 1) * (n - ii - jj - kk - 1) / n / (n - 1))
                        )

                    if jj > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii, jj - 1, kk + 1))
                        data.append(
                            2 * choose(n, 2) * (1.0 * (jj - 1) * (kk + 1) / n / (n - 1))
                        )

                    if kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii, jj + 1, kk - 1))
                        data.append(
                            2 * choose(n, 2) * (1.0 * (jj + 1) * (kk - 1) / n / (n - 1))
                        )

                    if kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii, jj, kk - 1))
                        data.append(
                            2
                            * choose(n, 2)
                            * (1.0 * (kk - 1) * (n - ii - jj - kk + 1) / n / (n - 1))
                        )

                    if n - ii - jj - kk > 0:
                        row.append(this_ind)
                        col.append(index_n(n, ii, jj, kk + 1))
                        data.append(
                            2
                            * choose(n, 2)
                            * (1.0 * (kk + 1) * (n - ii - jj - kk - 1) / n / (n - 1))
                        )

                    # outgoing density
                    row.append(this_ind)
                    col.append(this_ind)
                    data.append(
                        -2
                        * choose(n, 2)
                        * 2.0
                        * (
                            ii * (n - ii - jj - kk)
                            + ii * kk
                            + ii * jj
                            + jj * (n - ii - jj - kk)
                            + jj * kk
                            + kk * (n - ii - jj - kk)
                        )
                        / n
                        / (n - 1)
                    )

    return csc_matrix((data, (row, col)), shape=(Dsize, Dsize))


def mutations(n, theta=1.0):
    """
    Infinite-sites mutation model, where new mutations occur along the A/a and B/b
    backgrounds, with single-site SFS stored in [0,:,0] and [0,0,:].

    Note: might want to make theta accept a list of length two, which could be
    [theta_A, theta_B] in the future.

    :param n: The sample size.
    :param theta: The scaled mutation rate.
    """
    Msize = int((n + 1) * (n + 2) * (n + 3) / 6)

    # M_1to2 = np.zeros((Msize, Msize))
    M_1to2_entries = defaultdict(float)
    # M_1to2_entries[(row, col)] = val

    # A/a -> AB and aB
    M_1to2_entries = defaultdict(float)
    for j in range(0, n - 1):  # B falls on A background
        M_1to2_entries[(index_n(n, 1, j, 0), index_n(n, 0, j + 1, 0))] += (
            (j + 1) * theta / 2.0
        )
    for j in range(1, n):  # B falls on a background
        M_1to2_entries[(index_n(n, 0, j, 1), index_n(n, 0, j, 0))] += (
            (n - j) * theta / 2.0
        )
    # B/b -> AB and Ab
    for k in range(0, n - 1):
        M_1to2_entries[(index_n(n, 1, 0, k), index_n(n, 0, 0, k + 1))] += (
            (k + 1) * theta / 2.0
        )
    for k in range(1, n):
        M_1to2_entries[(index_n(n, 0, 1, k), index_n(n, 0, 0, k))] += (
            (n - k) * theta / 2.0
        )

    M_0to1 = np.zeros(Msize)
    M_0to1[index_n(n, 0, 0, 1)] = M_0to1[index_n(n, 0, 1, 0)] = n * theta / 2.0

    rows = []
    cols = []
    data = []
    for k, v in M_1to2_entries.items():
        rows.append(k[0])
        cols.append(k[1])
        data.append(v)
    M_1to2 = csc_matrix((data, (rows, cols)), shape=(Msize, Msize))

    return M_0to1, M_1to2


def mutations_mnm(n, theta=1.0):
    """
    Instead of mutations falling on the background of already segregating mutations,
    we assume clustered mutations occur, which create AB haplotypes directly, and
    don't depend on the single locus distribution.

    :param n: The sample size.
    :param theta: The scaled mutation rate.
    """
    Msize = int((n + 1) * (n + 2) * (n + 3) / 6)
    M_0to1 = np.zeros(Msize)
    M_0to1[index_n(n, 0, 0, 1)] = M_0to1[index_n(n, 0, 1, 0)] = n * theta / 2.0
    M_0to1[index_n(n, 1, 0, 0)] = n * theta / 2.0
    return M_0to1


def recombination(n, rho):
    """
    Returns the recombination operater for a given sample size.

    :param n: The sample size.
    :param rho: 4*Ne*r, where r is the recombination probability between two loci.
    """
    Rsize0 = int((n + 1) * (n + 2) * (n + 3) / 6)
    Rsize1 = int((n + 2) * (n + 3) * (n + 4) / 6)
    row = []
    col = []
    data = []

    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                fA = i + j
                fa = n - i - j
                fB = i + k
                fb = n - i - k
                if fA == 0 or fa == 0 or fB == 0 or fb == 0:
                    continue

                # incoming
                if j > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i + 1, j - 1, k))
                    data.append(rho / 2.0 * (i + 1) * (n - i - j - k + 1) / (n + 1))

                if k > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i + 1, j, k - 1))
                    data.append(rho / 2.0 * (i + 1) * (n - i - j - k + 1) / (n + 1))

                if i > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i - 1, j + 1, k + 1))
                    data.append(rho / 2.0 * (j + 1) * (k + 1) / (n + 1))

                if n - i - j - k > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i, j + 1, k + 1))
                    data.append(rho / 2.0 * (j + 1) * (k + 1) / (n + 1))

                # outgoing
                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i + 1, j, k))
                data.append(-rho / 2.0 * (i + 1) * (n - i - j - k) / (n + 1))

                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i, j + 1, k))
                data.append(-rho / 2.0 * (j + 1) * k / (n + 1))

                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i, j, k + 1))
                data.append(-rho / 2.0 * j * (k + 1) / (n + 1))

                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i, j, k))
                data.append(-rho / 2.0 * i * (n - i - j - k + 1) / (n + 1))

    return csc_matrix((data, (row, col)), shape=(Rsize0, Rsize1))


def selection_two_locus(n, sel_params):
    """
    This is for additive selection at both loci, where Ab has selection coefficient sA,
    aB has sB, and AB has sAB
    This is an additive model, meaning at each locus selection acts additively, but
    it allows for epistasis if sAB != sA + sB. (I.e. sAB = sA + sB + epsilon)

    :param n: The sample size.
    :param sel_params: The list of selection coefficients, where
        sel_params = [sAB, sA, sB].
    """
    sAB, sA, sB = sel_params
    Ssize0 = int((n + 1) * (n + 2) * (n + 3) / 6)
    Ssize1 = int((n + 2) * (n + 3) * (n + 4) / 6)

    row = []
    col = []
    data = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                this_ind = index_n(n, i, j, k)
                if i == 0 and j == 0 and k == 0:
                    continue
                if i + j == n:
                    continue
                if i + k == n:
                    continue

                if i + j == 0:
                    # nA = 0, 1 <= k <= n - 1
                    row.append(this_ind)
                    col.append(index_n(n + 1, i, j, k + 1))
                    data.append(-1.0 / (n + 1) * sB * (k + 1) * (n - k))
                    row.append(this_ind)
                    col.append(index_n(n + 1, i, j, k))
                    data.append(1.0 / (n + 1) * sB * k * (n - k + 1))
                    continue

                if i + k == 0:
                    # nB = 0, 1 <= j <= n - 1
                    row.append(this_ind)
                    col.append(index_n(n + 1, i, j + 1, k))
                    data.append(-1.0 / (n + 1) * sA * (j + 1) * (n - j))
                    row.append(this_ind)
                    col.append(index_n(n + 1, i, j, k))
                    data.append(1.0 / (n + 1) * sA * j * (n - j + 1))
                    continue

                row.append(this_ind)
                col.append(index_n(n + 1, i + 1, j, k))
                data.append(
                    1.0
                    / (n + 1)
                    * (-sAB * (i + 1) * (n - i) + sA * (i + 1) * j + sB * (i + 1) * k)
                )
                row.append(this_ind)
                col.append(index_n(n + 1, i, j + 1, k))
                data.append(
                    1.0
                    / (n + 1)
                    * (sAB * i * (j + 1) - sA * (j + 1) * (n - j) + sB * (j + 1) * k)
                )
                row.append(this_ind)
                col.append(index_n(n + 1, i, j, k + 1))
                data.append(
                    1.0
                    / (n + 1)
                    * (sAB * i * (k + 1) + sA * j * (k + 1) - sB * (k + 1) * (n - k))
                )
                row.append(this_ind)
                col.append(index_n(n + 1, i, j, k))
                data.append(
                    1.0
                    / (n + 1)
                    * (
                        sAB * i * (n - i - j - k + 1)
                        + sA * j * (n - i - j - k + 1)
                        + sB * k * (n - i - j - k + 1)
                    )
                )
    return csc_matrix((data, (row, col)), shape=(Ssize0, Ssize1))


def selection_general(n, sel_params):
    """
    A general selection operator, that works with a jackknife with a jump of two.
    The fitnesses (via selection coefficients) of each diploid two-locus genotype are:

     - ab/ab: 1
     - Ab/ab: 1 + s_Ab_ab
     - aB/ab: 1 + s_aB_ab
     - AB/ab: 1 + s_AB_ab
     - Ab/Ab: 1 + s_Ab_Ab
     - aB/Ab: 1 + s_Ab_aB
     - AB/Ab: 1 + s_AB_Ab
     - aB/aB: 1 + s_aB_aB
     - AB/aB: 1 + s_AB_aB
     - AB/AB: 1 + s_AB_AB

    There are some helper functions that convert typical selection scenarios
    (additivity and domininace, epistasis, etc) to these selection parameters.

    :param n: The sample size.
    :param sel_params: The selection parameters for each diploid genotype, as (s_AB_AB,
        s_AB_Ab, s_AB_aB, s_AB_ab, s_Ab_Ab, s_Ab_aB, s_Ab_ab, s_aB_aB, s_aB_ab)
    """
    (
        s_AB_AB,
        s_AB_Ab,
        s_AB_aB,
        s_AB_ab,
        s_Ab_Ab,
        s_Ab_aB,
        s_Ab_ab,
        s_aB_aB,
        s_aB_ab,
    ) = sel_params
    Ssize0 = int((n + 1) * (n + 2) * (n + 3) / 6)
    Ssize2 = int((n + 3) * (n + 4) * (n + 5) / 6)

    row = []
    col = []
    data = []

    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                this_ind = index_n(n, i, j, k)
                if i == 0 and j == 0 and k == 0:
                    continue
                if i + j == n:
                    continue
                if i + k == n:
                    continue

                if i + j == 0:
                    # nA = 0, 1 <= k <= n - 1
                    row.append(this_ind)
                    col.append(index_n(n + 2, i, j, k))
                    data.append(
                        1.0
                        / (n + 2)
                        / (n + 1)
                        * s_aB_ab
                        * k
                        * (n - k + 2)
                        * (n - k + 1)
                    )
                    row.append(this_ind)
                    col.append(index_n(n + 2, i, j, k + 1))
                    data.append(
                        1.0
                        / (n + 2)
                        / (n + 1)
                        * (
                            -s_aB_ab * (k + 1) * k * (n - k + 1)
                            - s_aB_ab * (k + 1) * (n - k + 1) * (n - k)
                            + s_aB_aB * (k + 1) * k * (n - k + 1)
                        )
                    )
                    row.append(this_ind)
                    col.append(index_n(n + 2, i, j, k + 2))
                    data.append(
                        1.0
                        / (n + 2)
                        / (n + 1)
                        * (
                            s_aB_ab * (k + 2) * (k + 1) * (n - k)
                            - s_aB_aB * (k + 2) * (k + 1) * (n - k)
                        )
                    )
                    continue

                if i + k == 0:
                    # nB = 0, 1 <= j <= n - 1
                    row.append(this_ind)
                    col.append(index_n(n + 2, i, j, k))
                    data.append(
                        1.0
                        / (n + 2)
                        / (n + 1)
                        * s_Ab_ab
                        * j
                        * (n - j + 2)
                        * (n - j + 1)
                    )
                    row.append(this_ind)
                    col.append(index_n(n + 2, i, j + 1, k))
                    data.append(
                        1.0
                        / (n + 2)
                        / (n + 1)
                        * (
                            -s_Ab_ab * (j + 1) * j * (n - j + 1)
                            - s_Ab_ab * (j + 1) * (n - j + 1) * (n - j)
                            + s_Ab_Ab * (j + 1) * j * (n - j + 1)
                        )
                    )
                    row.append(this_ind)
                    col.append(index_n(n + 2, i, j + 2, k))
                    data.append(
                        1.0
                        / (n + 2)
                        / (n + 1)
                        * (
                            s_Ab_ab * (j + 2) * (j + 1) * (n - j)
                            - s_Ab_Ab * (j + 2) * (j + 1) * (n - j)
                        )
                    )
                    continue

                row.append(this_ind)
                col.append(index_n(n + 2, i, j, k))
                data.append(
                    (n - i - j - k + 2)
                    * (n - i - j - k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (s_AB_ab * i + s_Ab_ab * j + s_aB_ab * k)
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i + 1, j, k))
                data.append(
                    (i + 1)
                    * (n - i - j - k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        -s_AB_ab * j
                        - s_AB_ab * k
                        - s_AB_ab * i
                        + s_Ab_ab * j
                        + s_AB_Ab * j
                        + s_aB_ab * k
                        + s_AB_aB * k
                        - s_AB_ab * (n - i - j - k)
                        - s_AB_ab * j
                        - s_AB_ab * k
                        + s_AB_AB * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i + 2, j, k))
                data.append(
                    (i + 2)
                    * (i + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        s_AB_ab * (n - i - j - k)
                        + s_AB_Ab * j
                        + s_AB_aB * k
                        - s_AB_AB * (n - i - j - k)
                        - s_AB_AB * j
                        - s_AB_AB * k
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i + 1, j + 1, k))
                data.append(
                    (i + 1)
                    * (j + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        s_Ab_ab * (n - i - j - k)
                        + s_AB_ab * (n - i - j - k)
                        + s_Ab_Ab * j
                        - s_AB_Ab * (n - i - j - k)
                        - s_AB_Ab * k
                        - s_AB_Ab * i
                        + s_Ab_aB * k
                        + s_AB_aB * k
                        - s_AB_Ab * (n - i - j - k)
                        - s_AB_Ab * j
                        - s_AB_Ab * k
                        + s_AB_AB * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i + 1, j, k + 1))
                data.append(
                    (i + 1)
                    * (k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        s_aB_ab * (n - i - j - k)
                        + s_AB_ab * (n - i - j - k)
                        + s_Ab_aB * j
                        + s_AB_Ab * j
                        + s_aB_aB * k
                        - s_AB_aB * (n - i - j - k)
                        - s_AB_aB * j
                        - s_AB_aB * i
                        - s_AB_aB * (n - i - j - k)
                        - s_AB_aB * j
                        - s_AB_aB * k
                        + s_AB_AB * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i, j + 1, k))
                data.append(
                    (j + 1)
                    * (n - i - j - k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        -s_Ab_ab * j
                        - s_Ab_ab * k
                        - s_Ab_ab * i
                        - s_Ab_ab * (n - i - j - k)
                        - s_Ab_ab * k
                        - s_Ab_ab * i
                        + s_Ab_Ab * j
                        + s_aB_ab * k
                        + s_Ab_aB * k
                        + s_AB_ab * i
                        + s_AB_Ab * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i, j + 2, k))
                data.append(
                    (j + 2)
                    * (j + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        s_Ab_ab * (n - i - j - k)
                        - s_Ab_Ab * (n - i - j - k)
                        - s_Ab_Ab * k
                        - s_Ab_Ab * i
                        + s_Ab_aB * k
                        + s_AB_Ab * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i, j + 1, k + 1))
                data.append(
                    (j + 1)
                    * (k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        s_Ab_ab * (n - i - j - k)
                        + s_aB_ab * (n - i - j - k)
                        + s_Ab_Ab * j
                        - s_Ab_aB * (n - i - j - k)
                        - s_Ab_aB * k
                        - s_Ab_aB * i
                        - s_Ab_aB * (n - i - j - k)
                        - s_Ab_aB * j
                        - s_Ab_aB * i
                        + s_aB_aB * k
                        + s_AB_Ab * i
                        + s_AB_aB * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i, j, k + 1))
                data.append(
                    (k + 1)
                    * (n - i - j - k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        -s_aB_ab * j
                        - s_aB_ab * k
                        - s_aB_ab * i
                        + s_Ab_ab * j
                        + s_Ab_aB * j
                        - s_aB_ab * (n - i - j - k)
                        - s_aB_ab * j
                        - s_aB_ab * i
                        + s_aB_aB * k
                        + s_AB_aB * i
                        + s_AB_ab * i
                    )
                )
                row.append(this_ind)
                col.append(index_n(n + 2, i, j, k + 2))
                data.append(
                    (k + 2)
                    * (k + 1)
                    / (n + 2)
                    / (n + 1)
                    * (
                        s_aB_ab * (n - i - j - k)
                        + s_Ab_aB * j
                        - s_aB_aB * (n - i - j - k)
                        - s_aB_aB * j
                        - s_aB_aB * i
                        + s_AB_aB * i
                    )
                )

    return csc_matrix((data, (row, col)), shape=(Ssize0, Ssize2))


# from dadi.TwoLocus, projecting the spectrum to smaller sample size
def ln_binomial(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


projection_cache = {}


def cached_projection(proj_to, proj_from, indexes, cache=True):
    """
    Coefficients for projection from a larger size to smaller. These are stored
    as a scipy sparse csr matrix, flattening to array form. This allows us to
    cache the projections without running into memory issues.

    :param proj_to: Number of samples to project down to
    :param proj_from: Number of samples to project from
    :param indexes: Number of derived alleles projecting from - tuple of (nAB, nAb, naB)
    :param cache: If True, cache intermediate results for faster projection. These are
        stored as sparse vectors in vector format, which can be converted to 3D
        spectrum format using Phi_to_array.
    """
    key = (proj_to, proj_from, indexes)
    try:
        return projection_cache[key]
    except KeyError:
        X1, X2, X3 = indexes
        X4 = proj_from - X1 - X2 - X3
        #proj_weights = np.zeros((proj_to + 1, proj_to + 1, proj_to + 1))
        #proj_weights = np.zeros(int((proj_to + 1) * (proj_to + 2) * (proj_to + 3) / 6))
        cols = []
        data = []
        for ii in range(X1 + 1):
            for jj in range(X2 + 1):
                for kk in range(X3 + 1):
                    ll = proj_to - ii - jj - kk
                    if ll > X4 or ll < 0:
                        continue
                    f = (
                        ln_binomial(X1, ii)
                        + ln_binomial(X2, jj)
                        + ln_binomial(X3, kk)
                        + ln_binomial(X4, ll)
                        - ln_binomial(proj_from, proj_to)
                    )
                    #proj_weights[index_n(ii, jj, kk] = np.exp(f)
                    #proj_weights[index_n(proj_to, ii, jj, kk)] = np.exp(f)
                    cols.append(index_n(proj_to, ii, jj, kk))
                    data.append(np.exp(f))
        
        rows = [0] * len(cols)
        size = int((proj_to + 1) * (proj_to + 2) * (proj_to + 3) / 6)
        proj_weights = csr_matrix((data, (rows, cols)), shape=(1, size))

        if cache:
            projection_cache[key] = proj_weights

        return proj_weights


def project(F_from, proj_to, cache=True):
    proj_from = F_from.sample_size
    if proj_to == proj_from:
        return F_from
    elif proj_to > proj_from:
        raise ValueError("Projection size must be smaller than spectrum size.")
    else:
        size = int((proj_to + 1) * (proj_to + 2) * (proj_to + 3) / 6)
        Phi_proj = csr_matrix((1, size))
        for X1 in range(proj_from):
            for X2 in range(proj_from):
                for X3 in range(proj_from):
                    if F_from.mask[X1, X2, X3] == True:
                        # don't project over masked entries
                        continue
                    elif F_from[X1, X2, X3] == 0:
                        # no density in this bin, can skip
                        continue
                    else:
                        hits = (X1, X2, X3)
                        proj_weights = cached_projection(
                            proj_to, proj_from, hits, cache=cache
                        )
                        Phi_proj += proj_weights * F_from[X1, X2, X3]
        # handle the A/a and B/b edges separately
        F_proj_2loc = Phi_to_array(Phi_proj.toarray()[0], proj_to)
        F_proj_2loc[0, :, 0] = F_proj_2loc[0, 0, :] = 0

        Phi_proj_A = csr_matrix((1, size))
        for X2 in range(1, proj_from):
            hits = (0, X2, 0)
            proj_weights = cached_projection(proj_to, proj_from, hits, cache=cache)
            Phi_proj_A += proj_weights * F_from[0, X2, 0]

        Phi_proj_B = csr_matrix((1, size))
        for X3 in range(1, proj_from):
            hits = (0, 0, X3)
            proj_weights = cached_projection(proj_to, proj_from, hits, cache=cache)
            Phi_proj_B += proj_weights * F_from[0, 0, X3]
        
        F_proj_A = Phi_to_array(Phi_proj_A.toarray()[0], proj_to)
        F_proj_B = Phi_to_array(Phi_proj_B.toarray()[0], proj_to)

        F_proj = F_proj_2loc + F_proj_A + F_proj_B
        return F_proj


## methods for reversible mutation model


def mutations_reversible(n, u, v):
    """
    Assuming equal forward and backward mutation rates, but allowing different rates
    at left (u) and right (v) loci
    """
    Msize = int((n + 1) * (n + 2) * (n + 3) / 6)

    M = np.zeros((Msize, Msize))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                this_ind = index_n(n, i, j, k)
                if i > 0:
                    M[this_ind, index_n(n, i - 1, j, k + 1)] += u * (k + 1)
                    M[this_ind, index_n(n, i - 1, j + 1, k)] += v * (j + 1)
                if j > 0:
                    M[this_ind, index_n(n, i, j - 1, k)] += u * (n - i - j - k + 1)
                    M[this_ind, index_n(n, i + 1, j - 1, k)] += v * (i + 1)
                if k > 0:
                    M[this_ind, index_n(n, i + 1, j, k - 1)] += u * (i + 1)
                    M[this_ind, index_n(n, i, j, k - 1)] += v * (n - i - j - k + 1)
                if n - i - j - k > 0:
                    M[this_ind, index_n(n, i, j + 1, k)] += u * (j + 1)
                    M[this_ind, index_n(n, i, j, k + 1)] += v * (k + 1)

                M[this_ind, this_ind] -= (u + v) * n

    return csc_matrix(M)


def mutations_reversible_2(n, u, v):
    """
    we allow only mutations if the frequency is zero
    if a mutation fixes, put that density back at sero
    """
    Msize = int((n + 1) * (n + 2) * (n + 3) / 6)

    M = np.zeros((Msize, Msize))

    # fA = i+j
    i = 0
    j = 0
    # mutations introduce new A mutation along aB/ab axis
    # B/b -> AB and Ab
    for k in range(0, n - 1):
        M[index_n(n, 1, 0, k), index_n(n, 0, 0, k + 1)] += (k + 1) * u / 2.0
        M[index_n(n, 0, 0, k + 1), index_n(n, 0, 0, k + 1)] -= (k + 1) * u / 2.0
    for k in range(0, n - 1):
        M[index_n(n, 0, 1, k), index_n(n, 0, 0, k)] += (n - k) * u / 2.0
        M[index_n(n, 0, 0, k), index_n(n, 0, 0, k)] -= (n - k) * u / 2.0

    # fB = i+k
    i = 0
    k = 0
    # mutations introduce new A mutation along Ab/ab
    # A/a -> AB and aB
    for j in range(0, n - 1):
        M[index_n(n, 1, j, 0), index_n(n, 0, j + 1, 0)] += (j + 1) * v / 2.0
        M[index_n(n, 0, j + 1, 0), index_n(n, 0, j + 1, 0)] -= (j + 1) * v / 2.0
    for j in range(0, n - 1):
        M[index_n(n, 0, j, 1), index_n(n, 0, j, 0)] += (n - j) * v / 2.0
        M[index_n(n, 0, j, 0), index_n(n, 0, j, 0)] -= (n - j) * v / 2.0

    # return fixed density to origin
    M2 = np.zeros((Msize, Msize))
    M2[index_n(n, 0, 0, 0), index_n(n, n, 0, 0)] += 1
    M2[index_n(n, n, 0, 0), index_n(n, n, 0, 0)] -= 1
    M2[index_n(n, 0, 0, 0), index_n(n, 0, n, 0)] += 1
    M2[index_n(n, 0, n, 0), index_n(n, 0, n, 0)] -= 1
    M2[index_n(n, 0, 0, 0), index_n(n, 0, 0, n)] += 1
    M2[index_n(n, 0, 0, n), index_n(n, 0, 0, n)] -= 1
    return csc_matrix(M), M2


def drift_reversible(n):
    Dsize = int((n + 1) * (n + 2) * (n + 3) / 6)
    row = []
    col = []
    data = []
    for ii in range(n + 1):
        for jj in range(n + 1 - ii):
            for kk in range(n + 1 - ii - jj):
                this_ind = index_n(n, ii, jj, kk)
                # incoming density
                if ii > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii - 1, jj, kk))
                    data.append(
                        2
                        * choose(n, 2)
                        * (1.0 * (ii - 1) * (n - ii - jj - kk + 1) / n / (n - 1))
                    )

                if n - ii - jj - kk > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii + 1, jj, kk))
                    data.append(
                        2
                        * choose(n, 2)
                        * (1.0 * (ii + 1) * (n - ii - jj - kk - 1) / n / (n - 1))
                    )

                if ii > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii - 1, jj, kk + 1))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (ii - 1) * (kk + 1) / n / (n - 1))
                    )

                if kk > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii + 1, jj, kk - 1))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (ii + 1) * (kk - 1) / n / (n - 1))
                    )

                if ii > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii - 1, jj + 1, kk))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (ii - 1) * (jj + 1) / n / (n - 1))
                    )

                if jj > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii + 1, jj - 1, kk))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (ii + 1) * (jj - 1) / n / (n - 1))
                    )

                if jj > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj - 1, kk))
                    data.append(
                        2
                        * choose(n, 2)
                        * (1.0 * (jj - 1) * (n - ii - jj - kk + 1) / n / (n - 1))
                    )

                if n - ii - jj - kk > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj + 1, kk))
                    data.append(
                        2
                        * choose(n, 2)
                        * (1.0 * (jj + 1) * (n - ii - jj - kk - 1) / n / (n - 1))
                    )

                if jj > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj - 1, kk + 1))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (jj - 1) * (kk + 1) / n / (n - 1))
                    )

                if kk > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj + 1, kk - 1))
                    data.append(
                        2 * choose(n, 2) * (1.0 * (jj + 1) * (kk - 1) / n / (n - 1))
                    )

                if kk > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj, kk - 1))
                    data.append(
                        2
                        * choose(n, 2)
                        * (1.0 * (kk - 1) * (n - ii - jj - kk + 1) / n / (n - 1))
                    )

                if n - ii - jj - kk > 0:
                    row.append(this_ind)
                    col.append(index_n(n, ii, jj, kk + 1))
                    data.append(
                        2
                        * choose(n, 2)
                        * (1.0 * (kk + 1) * (n - ii - jj - kk - 1) / n / (n - 1))
                    )

                # outgoing density
                row.append(this_ind)
                col.append(this_ind)
                data.append(
                    -2
                    * choose(n, 2)
                    * 2.0
                    * (
                        ii * (n - ii - jj - kk)
                        + ii * kk
                        + ii * jj
                        + jj * (n - ii - jj - kk)
                        + jj * kk
                        + kk * (n - ii - jj - kk)
                    )
                    / n
                    / (n - 1)
                )

    return csc_matrix((data, (row, col)), shape=(Dsize, Dsize))


def recombination_reversible(n, rho):
    """
    rho = 4*Ne*r
    where r is the recombination probability
    """
    Rsize0 = int((n + 1) * (n + 2) * (n + 3) / 6)
    Rsize1 = int((n + 2) * (n + 3) * (n + 4) / 6)
    row = []
    col = []
    data = []

    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                fA = i + j
                fa = n - i - j
                fB = i + k
                fb = n - i - k
                # incoming
                if j > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i + 1, j - 1, k))
                    data.append(
                        n
                        * rho
                        / 2.0
                        * 1.0
                        * (i + 1)
                        * (n - i - j - k + 1)
                        / (n + 1)
                        / n
                    )

                if k > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i + 1, j, k - 1))
                    data.append(
                        n
                        * rho
                        / 2.0
                        * 1.0
                        * (i + 1)
                        * (n - i - j - k + 1)
                        / (n + 1)
                        / n
                    )

                if i > 0:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i - 1, j + 1, k + 1))
                    data.append(n * rho / 2.0 * 1.0 * (j + 1) * (k + 1) / (n + 1) / n)

                if i + j + k + 1 < n + 1:
                    row.append(index_n(n, i, j, k))
                    col.append(index_n(n + 1, i, j + 1, k + 1))
                    data.append(n * rho / 2.0 * 1.0 * (j + 1) * (k + 1) / (n + 1) / n)

                # outgoing
                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i + 1, j, k))
                data.append(
                    -n * rho / 2.0 * 1.0 * (i + 1) * (n - i - j - k) / (n + 1) / n
                )

                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i, j + 1, k))
                data.append(-n * rho / 2.0 * 1.0 * (j + 1) * (k) / (n + 1) / n)

                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i, j, k + 1))
                data.append(-n * rho / 2.0 * 1.0 * (j) * (k + 1) / (n + 1) / n)

                row.append(index_n(n, i, j, k))
                col.append(index_n(n + 1, i, j, k))
                data.append(
                    -n * rho / 2.0 * 1.0 * (i) * (n - i - j - k + 1) / (n + 1) / n
                )

    return csc_matrix((data, (row, col)), shape=(Rsize0, Rsize1))


def selection_reversible_additive(n):
    """
    selection at just the left locus, accounting for selection at fixed entries as well
    for now, just additive (n->n+1)
    """
    Ssize0 = int((n + 1) * (n + 2) * (n + 3) / 6)
    Ssize1 = int((n + 2) * (n + 3) * (n + 4) / 6)
    S = np.zeros((Ssize0, Ssize1))
    for ii in range(n + 1):
        for jj in range(n + 1 - ii):
            for kk in range(n + 1 - ii - jj):
                this_ind = index_n(n, ii, jj, kk)
                if ii > 0:
                    S[this_ind, index_n(n + 1, ii, jj + 1, kk)] += (
                        ii * (jj + 1.0) / (n + 1.0)
                    )
                    S[
                        index_n(n, ii - 1, jj + 1, kk), index_n(n + 1, ii, jj + 1, kk)
                    ] -= (ii * (jj + 1.0) / (n + 1.0))

                    S[this_ind, index_n(n + 1, ii, jj, kk + 1)] += (
                        ii * (kk + 1.0) / (n + 1.0)
                    )
                    S[
                        index_n(n, ii - 1, jj, kk + 1), index_n(n + 1, ii, jj, kk + 1)
                    ] -= (ii * (kk + 1.0) / (n + 1.0))

                    S[this_ind, index_n(n + 1, ii, jj, kk)] += (
                        ii * (n - ii - jj - kk + 1.0) / (n + 1.0)
                    )
                    S[index_n(n, ii - 1, jj, kk), index_n(n + 1, ii, jj, kk)] -= (
                        ii * (n - ii - jj - kk + 1.0) / (n + 1.0)
                    )
                if jj > 0:
                    S[this_ind, index_n(n + 1, ii + 1, jj, kk)] += (
                        jj * (ii + 1.0) / (n + 1.0)
                    )
                    S[
                        index_n(n, ii + 1, jj - 1, kk), index_n(n + 1, ii + 1, jj, kk)
                    ] -= (jj * (ii + 1.0) / (n + 1.0))

                    S[this_ind, index_n(n + 1, ii, jj, kk + 1)] += (
                        jj * (kk + 1.0) / (n + 1.0)
                    )
                    S[
                        index_n(n, ii, jj - 1, kk + 1), index_n(n + 1, ii, jj, kk + 1)
                    ] -= (jj * (kk + 1.0) / (n + 1.0))

                    S[this_ind, index_n(n + 1, ii, jj, kk)] += (
                        jj * (n - ii - jj - kk + 1.0) / (n + 1.0)
                    )
                    S[index_n(n, ii, jj - 1, kk), index_n(n + 1, ii, jj, kk)] -= (
                        jj * (n - ii - jj - kk + 1.0) / (n + 1.0)
                    )

    return csc_matrix(S)
