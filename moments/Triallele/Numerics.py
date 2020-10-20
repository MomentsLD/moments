import numpy as np, math
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from scipy.sparse import identity

"""
Numerics file for Triallele frequency spectrum model
"""


def flatten(T):
    """
    takes 2D spectrum T and flattens into array Phi, indexed by index_n
    """
    ns = len(T) - 1
    Phi = np.zeros(int((ns + 1) * (ns + 2) / 2))
    for ii in range(ns + 1):
        start = sum(range(ns - ii + 2, ns + 2))
        Phi[start : start + ns - ii + 1] = T[ii, : ns - ii + 1]
    return Phi


def reform(Phi, ns):
    T = np.zeros((ns + 1, ns + 1))
    for ii in range(ns + 1):
        start = sum(range(ns - ii + 2, ns + 2))
        T[ii, : ns - ii + 1] = Phi[start : start + ns - ii + 1]
    return T


def index_n(ns):
    indexes = {}
    for ii in range(ns + 1):
        start = sum(range(ns - ii + 2, ns + 2))
        for jj in range(ns - ii + 1):
            indexes[(ii, jj)] = start + jj
    return indexes


index_cache = {}


def get_index(ns, ii, jj):
    try:
        return index_cache[ns][(ii, jj)]
    except KeyError:
        index_cache.setdefault(ns, {})
        index_cache[ns] = index_n(ns)
        return index_cache[ns][(ii, jj)]


def choose(n, i):
    return np.exp(gammaln(n + 1) - gammaln(n - i + 1) - gammaln(i + 1))


def drift(ns):
    Dsize = int((ns + 1) * (ns + 2) / 2)
    row = []
    col = []
    data = []
    for ii in range(ns + 1):
        for jj in range(ns + 1 - ii):
            if (
                (ii == 0 and jj == 0)
                or (ii == 0 and jj == ns)
                or (jj == 0 and ii == ns)
            ):
                continue
            this_ind = get_index(ns, ii, jj)
            # incoming density
            if ii + jj < ns:
                if ii == 0:
                    # D[this_ind, get_index(ns,ii,jj-1)] = 2*choose(ns,2)*(1.*(jj-1)*(ns-jj+1)/ns/(ns-1))
                    row.append(this_ind)
                    col.append(get_index(ns, ii, jj - 1))
                    data.append(
                        2
                        * choose(ns, 2)
                        * (1.0 * (jj - 1) * (ns - jj + 1) / ns / (ns - 1))
                    )
                    # D[this_ind, get_index(ns,ii,jj)] = 2*choose(ns,2)*(-2.*jj*(ns-jj)/ns/(ns-1))
                    row.append(this_ind)
                    col.append(get_index(ns, ii, jj))
                    data.append(
                        2 * choose(ns, 2) * (-2.0 * jj * (ns - jj) / ns / (ns - 1))
                    )
                    # D[this_ind, get_index(ns,ii,jj+1)] = 2*choose(ns,2)*(1.*(jj+1)*(ns-jj-1)/ns/(ns-1))
                    row.append(this_ind)
                    col.append(get_index(ns, ii, jj + 1))
                    data.append(
                        2
                        * choose(ns, 2)
                        * (1.0 * (jj + 1) * (ns - jj - 1) / ns / (ns - 1))
                    )
                elif jj == 0:
                    # D[this_ind, get_index(ns,ii-1,jj)] = 2*choose(ns,2)*(1.*(ii-1)*(ns-ii+1)/ns/(ns-1))
                    row.append(this_ind)
                    col.append(get_index(ns, ii - 1, jj))
                    data.append(
                        2
                        * choose(ns, 2)
                        * (1.0 * (ii - 1) * (ns - ii + 1) / ns / (ns - 1))
                    )
                    # D[this_ind, get_index(ns,ii,jj)] = 2*choose(ns,2)*(-2.*ii*(ns-ii)/ns/(ns-1))
                    row.append(this_ind)
                    col.append(get_index(ns, ii, jj))
                    data.append(
                        2 * choose(ns, 2) * (-2.0 * ii * (ns - ii) / ns / (ns - 1))
                    )
                    # D[this_ind, get_index(ns,ii+1,jj)] = 2*choose(ns,2)*(1.*(ii+1)*(ns-ii-1)/ns/(ns-1))
                    row.append(this_ind)
                    col.append(get_index(ns, ii + 1, jj))
                    data.append(
                        2
                        * choose(ns, 2)
                        * (1.0 * (ii + 1) * (ns - ii - 1) / ns / (ns - 1))
                    )
                else:
                    if ii >= 1:
                        # D[this_ind, get_index(ns,ii-1,jj)] = 2*choose(ns,2) * (ii-1)*(ns-ii-jj+1)/float(ns*(ns-1))
                        row.append(this_ind)
                        col.append(get_index(ns, ii - 1, jj))
                        data.append(
                            2
                            * choose(ns, 2)
                            * (ii - 1)
                            * (ns - ii - jj + 1)
                            / float(ns * (ns - 1))
                        )
                        # D[this_ind, get_index(ns,ii-1,jj+1)] = 2*choose(ns,2) * (ii-1)*(jj+1)/float(ns*(ns-1))
                        row.append(this_ind)
                        col.append(get_index(ns, ii - 1, jj + 1))
                        data.append(
                            2
                            * choose(ns, 2)
                            * (ii - 1)
                            * (jj + 1)
                            / float(ns * (ns - 1))
                        )
                    if jj >= 1:
                        # D[this_ind, get_index(ns,ii,jj-1)] = 2*choose(ns,2) * (jj-1)*(ns-ii-jj+1)/float(ns*(ns-1))
                        row.append(this_ind)
                        col.append(get_index(ns, ii, jj - 1))
                        data.append(
                            2
                            * choose(ns, 2)
                            * (jj - 1)
                            * (ns - ii - jj + 1)
                            / float(ns * (ns - 1))
                        )
                        # D[this_ind, get_index(ns,ii+1,jj-1)] = 2*choose(ns,2) * (ii+1)*(jj-1)/float(ns*(ns-1))
                        row.append(this_ind)
                        col.append(get_index(ns, ii + 1, jj - 1))
                        data.append(
                            2
                            * choose(ns, 2)
                            * (ii + 1)
                            * (jj - 1)
                            / float(ns * (ns - 1))
                        )
                    if ns - ii - jj >= 1:
                        # D[this_ind, get_index(ns,ii+1,jj)] = 2*choose(ns,2) * (ii+1)*(ns-ii-jj-1)/float(ns*(ns-1))
                        row.append(this_ind)
                        col.append(get_index(ns, ii + 1, jj))
                        data.append(
                            2
                            * choose(ns, 2)
                            * (ii + 1)
                            * (ns - ii - jj - 1)
                            / float(ns * (ns - 1))
                        )
                        # D[this_ind, get_index(ns,ii,jj+1)] = 2*choose(ns,2) * (jj+1)*(ns-ii-jj-1)/float(ns*(ns-1))
                        row.append(this_ind)
                        col.append(get_index(ns, ii, jj + 1))
                        data.append(
                            2
                            * choose(ns, 2)
                            * (jj + 1)
                            * (ns - ii - jj - 1)
                            / float(ns * (ns - 1))
                        )
                    # outgoing density
                    row.append(this_ind)
                    col.append(this_ind)
                    data.append(
                        -2
                        * choose(ns, 2)
                        * (
                            2 * ii * (ns - ii - jj) / float(ns * (ns - 1))
                            + 2 * jj * (ns - ii - jj) / float(ns * (ns - 1))
                            + 2 * ii * jj / float(ns * (ns - 1))
                        )
                    )
                    # D[this_ind,this_ind] -= 2*choose(ns,2) * 2*ii*(ns-ii-jj)/float(ns*(ns-1))
                    # D[this_ind,this_ind] -= 2*choose(ns,2) * 2*jj*(ns-ii-jj)/float(ns*(ns-1))
                    # D[this_ind,this_ind] -= 2*choose(ns,2) * 2*ii*jj/float(ns*(ns-1))
    return csc_matrix((data, (row, col)), shape=(Dsize, Dsize))


def mutation(ns, theta0=1, theta1=1, theta2=1):
    """
    mutations for the infinite sites type model
    theta0 - scaled mutation rate for biallelic background
    theta1 - scaled mutation rate for allele 1 (against derived allele 2 background)
    theta2 - scaled mutation rate for allele 2 (against derived allele 1 background)
    """
    Bsize = int((ns + 1) * (ns + 2) / 2)
    # B_tri depends on background biallelic frequencies, so requires a dot product
    B_tri = np.zeros((Bsize, Bsize))
    # B_bi just gets added at rate dt*B_bi
    B_bi = np.zeros(Bsize)
    B_bi[get_index(ns, 0, 1)] = B_bi[get_index(ns, 1, 0)] = ns * theta0 / 2
    # mutations from allele 1 background to give rise to novel allele 2
    for ii in range(1, ns - 1):
        B_tri[get_index(ns, ii, 1), get_index(ns, ii, 0)] = (
            1.0 * theta2 / 2 * ns * (ns - ii) / ns
        )
        B_tri[get_index(ns, ii, 1), get_index(ns, ii + 1, 0)] = (
            1 * theta2 / 2 * ns * (ii + 1) / ns
        )

    for jj in range(1, ns - 1):
        B_tri[get_index(ns, 1, jj), get_index(ns, 0, jj)] = (
            1.0 * theta1 / 2 * ns * (ns - jj) / ns
        )
        B_tri[get_index(ns, 1, jj), get_index(ns, 0, jj + 1)] = (
            1 * theta1 / 2 * ns * (jj + 1) / ns
        )

    return B_bi, csc_matrix(B_tri)


def ln_binomial(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


projection_cache = {}


def cached_projection(proj_to, proj_from, hits):
    """
    Coefficients for projection from a larger size to smaller
    proj_to: Number of samples to project down to
    proj_from: Number of samples to project from
    hits: Number of derived alleles projecting from - tuple of (n1,n3)
    """
    key = (proj_to, proj_from, hits)
    try:
        return projection_cache[key]
    except KeyError:
        pass

    X1, X2 = hits
    X3 = proj_from - X1 - X2
    proj_weights = np.zeros((proj_to + 1, proj_to + 1))
    for ii in range(X1 + 1):
        for jj in range(X2 + 1):
            kk = proj_to - ii - jj
            if kk > X3 or kk < 0:
                continue
            f = np.exp(
                ln_binomial(X1, ii)
                + ln_binomial(X2, jj)
                + ln_binomial(X3, kk)
                - ln_binomial(proj_from, proj_to)
            )
            proj_weights[ii, jj] = f

    projection_cache[key] = proj_weights
    return proj_weights


## To do: improve projection to allow for finite genome projection!!!
def project(F_from, proj_to):
    proj_from = len(F_from) - 1
    if proj_to == proj_from:
        print("Projection to same size does not change anything, of course")
        return F_from
    elif proj_to > proj_from:
        raise ValueError("Projection must be to smaller size!")
    else:
        F_proj = np.zeros((proj_to + 1, proj_to + 1))
        # project interior points
        for X1 in range(proj_from + 1):
            for X2 in range(proj_from + 1 - X1):
                if X1 == 0 or X2 == 0:
                    continue
                hits = (X1, X2)
                proj_weights = cached_projection(proj_to, proj_from, hits)
                F_proj += proj_weights * F_from[X1, X2]
        F_proj[:, 0] = F_proj[0, :] = 0
        # project the edges
        for X1 in range(proj_from + 1):
            for X2 in range(proj_from + 1 - X1):
                if X1 == 0 or X2 == 0:
                    hits = (X1, X2)
                    proj_weights = cached_projection(proj_to, proj_from, hits)
                    F_proj += proj_weights * F_from[X1, X2]
        F_proj[0, 0] = 0
        return F_proj


def fold(F):
    folded = np.zeros(np.shape(F))
    ns = len(folded) - 1
    for ii in range(ns + 1):
        for jj in range(ns + 1 - ii):
            kk = ns - ii - jj
            if ii <= jj <= kk:  # minor alleles all ok, order [0,1,2]
                folded[ii, jj] += F[ii, jj]
            elif ii <= kk <= jj:  # [0,2,1]
                folded[ii, kk] += F[ii, jj]
            elif jj <= ii <= kk:  # [1,0,2]
                folded[jj, ii] += F[ii, jj]
            elif jj <= kk <= ii:  # [1,2,0]
                folded[jj, kk] += F[ii, jj]
            elif kk <= ii <= jj:  # [2,0,1]
                folded[kk, ii] += F[ii, jj]
            elif kk <= jj <= ii:  # [2,1,0]
                folded[kk, jj] += F[ii, jj]
    return folded


def selection(n, sel_params):
    sAA, sA0, sBB, sB0, sAB = sel_params
    Ssize_from = int((n + 1) * (n + 2) / 2)
    Ssize_to = int((n + 3) * (n + 4) / 2)
    S = np.zeros((Ssize_from, Ssize_to))
    ## selection transitions for triallelic sites
    for i in range(1, n):
        for j in range(1, n - i):
            # outgong
            # incoming

            # (AA) O
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] -= (
                -n
                * sAA
                / 3.0
                * choose(i + 1, 2)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 2, j)] += (
                -n
                * sAA
                / 3.0
                * choose(i + 2, 2)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )

            # (AA) B
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] -= (
                -n * sAA / 3.0 * choose(i + 1, 2) * choose(j + 1, 1) / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 2, j)] += (
                -n * sAA / 3.0 * choose(i + 2, 2) * choose(j, 1) / choose(n + 2, 3)
            )

            # (AO) A
            S[get_index(n, i, j), get_index(n + 2, i + 2, j)] -= (
                -n
                * sA0
                / 3.0
                * choose(i + 2, 2)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] += (
                -n
                * sA0
                / 3.0
                * choose(i + 1, 2)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )

            # (AO) B
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] -= (
                -n
                * sA0
                / 6.0
                * choose(i + 1, 1)
                * choose(n - i - j, 1)
                * choose(j + 1, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] += (
                -n
                * sA0
                / 6.0
                * choose(i + 1, 1)
                * choose(n - i - j + 1, 1)
                * choose(j, 1)
                / choose(n + 2, 3)
            )

            # (OA) O
            S[get_index(n, i, j), get_index(n + 2, i, j)] -= (
                -n
                * sA0
                / 3.0
                * choose(i, 1)
                * choose(n - i - j + 2, 2)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] += (
                -n
                * sA0
                / 3.0
                * choose(i + 1, 1)
                * choose(n - i - j + 1, 2)
                / choose(n + 2, 3)
            )

            # (OA) B
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] -= (
                -n
                * sA0
                / 6.0
                * choose(i, 1)
                * choose(j + 1, 1)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] += (
                -n
                * sA0
                / 6.0
                * choose(i + 1, 1)
                * choose(j, 1)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )

            # (BB) O
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] -= (
                -n
                * sBB
                / 3.0
                * choose(j + 1, 2)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i, j + 2)] += (
                -n
                * sBB
                / 3.0
                * choose(j + 2, 2)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )

            # (BB) A
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] -= (
                -n * sBB / 3.0 * choose(j + 1, 2) * choose(i + 1, 1) / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i, j + 2)] += (
                -n * sBB / 3.0 * choose(j + 2, 2) * choose(i, 1) / choose(n + 2, 3)
            )

            # (BO) A
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] -= (
                -n
                * sB0
                / 6.0
                * choose(i + 1, 1)
                * choose(j + 1, 1)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] += (
                -n
                * sB0
                / 6.0
                * choose(i, 1)
                * choose(j + 1, 1)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )

            # (BO) B
            S[get_index(n, i, j), get_index(n + 2, i, j + 2)] -= (
                -n
                * sB0
                / 3.0
                * choose(j + 2, 2)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] += (
                -n
                * sB0
                / 3.0
                * choose(j + 1, 2)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )

            # (OB) O
            S[get_index(n, i, j), get_index(n + 2, i, j)] -= (
                -n
                * sB0
                / 3.0
                * choose(j, 1)
                * choose(n - i - j + 2, 2)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] += (
                -n
                * sB0
                / 3.0
                * choose(j + 1, 1)
                * choose(n - i - j + 1, 2)
                / choose(n + 2, 3)
            )

            # (OB) A
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] -= (
                -n
                * sB0
                / 6.0
                * choose(i + 1, 1)
                * choose(j, 1)
                * choose(n - i - j + 1, 2)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] += (
                -n
                * sB0
                / 6.0
                * choose(i, 1)
                * choose(j + 1, 1)
                * choose(n - i - j + 1, 2)
                / choose(n + 2, 3)
            )

            # (BA) O
            S[get_index(n, i, j), get_index(n + 2, i, j + 1)] -= (
                -n
                * sAB
                / 6.0
                * choose(i, 1)
                * choose(j + 1, 1)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] += (
                -n
                * sAB
                / 6.0
                * choose(i + 1, 1)
                * choose(j + 1, 1)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )

            # (BA) B
            S[get_index(n, i, j), get_index(n + 2, i, j + 2)] -= (
                -n * sAB / 3.0 * choose(i, 1) * choose(j + 2, 2) / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] += (
                -n * sAB / 3.0 * choose(i + 1, 1) * choose(j + 1, 2) / choose(n + 2, 3)
            )

            # (AB) O
            S[get_index(n, i, j), get_index(n + 2, i + 1, j)] -= (
                -n
                * sAB
                / 6.0
                * choose(i + 1, 1)
                * choose(j, 1)
                * choose(n - i - j + 1, 1)
                / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] += (
                -n
                * sAB
                / 6.0
                * choose(i + 1, 1)
                * choose(j + 1, 1)
                * choose(n - i - j, 1)
                / choose(n + 2, 3)
            )

            # (AB) A
            S[get_index(n, i, j), get_index(n + 2, i, j + 2)] -= (
                -n * sAB / 3.0 * choose(i, 1) * choose(j + 2, 2) / choose(n + 2, 3)
            )
            S[get_index(n, i, j), get_index(n + 2, i + 1, j + 1)] += (
                -n * sAB / 3.0 * choose(i + 1, 1) * choose(j + 1, 2) / choose(n + 2, 3)
            )

    ## selection transitions for biallelic sites
    for i in range(n + 1)[1:n]:
        this_ind = get_index(n, i, 0)
        S[this_ind, get_index(n + 2, i + 2, 0)] += (
            -n * sAA / 3.0 * choose(i + 2, 2) * choose(n - i, 1) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, i + 1, 0)] += (
            -n * sA0 / 3.0 * choose(i + 1, 1) * choose(n - i + 1, 2) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, i + 1, 0)] += (
            -n * sA0 / 3.0 * choose(i + 1, 2) * choose(n - i + 1, 1) / choose(n + 2, 3)
        )

        S[this_ind, get_index(n + 2, i + 1, 0)] -= (
            -n * sAA / 3.0 * choose(i + 1, 2) * choose(n - i + 1, 1) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, i, 0)] -= (
            -n * sA0 / 3.0 * choose(i, 1) * choose(n - i + 2, 2) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, i + 2, 0)] -= (
            -n * sA0 / 3.0 * choose(i + 2, 2) * choose(n - i, 1) / choose(n + 2, 3)
        )

    for j in range(n + 1)[1:n]:
        this_ind = get_index(n, 0, j)
        S[this_ind, get_index(n + 2, 0, j + 2)] += (
            -n * sBB / 3.0 * choose(j + 2, 2) * choose(n - j, 1) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, 0, j + 1)] += (
            -n * sB0 / 3.0 * choose(j + 1, 1) * choose(n - j + 1, 2) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, 0, j + 1)] += (
            -n * sB0 / 3.0 * choose(j + 1, 2) * choose(n - j + 1, 1) / choose(n + 2, 3)
        )

        S[this_ind, get_index(n + 2, 0, j + 1)] -= (
            -n * sBB / 3.0 * choose(j + 1, 2) * choose(n - j + 1, 1) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, 0, j)] -= (
            -n * sB0 / 3.0 * choose(j, 1) * choose(n - j + 2, 2) / choose(n + 2, 3)
        )
        S[this_ind, get_index(n + 2, 0, j + 2)] -= (
            -n * sB0 / 3.0 * choose(j + 2, 2) * choose(n - j, 1) / choose(n + 2, 3)
        )

    return csc_matrix(S)
