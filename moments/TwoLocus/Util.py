import numpy as np


def pAB(F, nA, nB):
    """
    Returns the possible counts and frequencies of nAB haplotypes,
    given allele counts nA and nB at the left and right loci, resp.
    """
    n = F.sample_size
    if nA < 0 or nA > n:
        raise ValueError(f"nA must be between 0 and n={n}")
    if nB < 0 or nB > n:
        raise ValueError(f"nB must be between 0 and n={n}")
    min_AB = max(0, nA + nB - n)
    max_AB = min(nA, nB)
    counts = np.arange(min_AB, max_AB + 1)
    AB = np.zeros(len(counts))
    for idx, i in enumerate(counts):
        AB[idx] = F[i, nA - i, nB - i]
    return counts, AB


def additive_epistasis(s, epsilon=0, Ne=None):
    """
    Returns selection parameters [s_AB, s_A, s_B] where if Ne is not None, we assume
    we've been passed raw selection coefficients, so gamma = 2*Ne*s.

    With Ne given, returns [2 * gamma * (1 + epsilon), gamma, gamma]
    Withou Ne, returns [2 * s * (1+epsilon), s, s].

    In this model, epsilon of 0 is no epistasis, epsilon > 0 is synergistic epistasis,
    while epsilon < 0 is antagonistic epistasis. Note that epsilon < -1 will flip the
    sign of the selection coefficient for AB.
    """
    if Ne is None:
        gamma = s
    else:
        if Ne <= 0:
            raise ValueError("Ne must be positive")
        gamma = 2 * Ne * s

    return [2 * gamma * (1 + epsilon), gamma, gamma]


def simple_dominance(s, h=0.5, Ne=None):
    """
    Returns the general selection parameters for the simple dominance model with
    a single selection and dominance coefficient that applies to both loci:
    [s_AB_AB, s_AB_Ab, s_AB_aB, s_AB_ab, s_Ab_Ab, s_Ab_aB, s_Ab_ab, s_aB_aB, s_aB_ab]

    Selection is multiplicative across loci, and we assume sA and sB are small, so
    that the cross term sA * sB can be ignored. Very strong selection will violate
    this assumption.
    """
    if Ne is None:
        gamma = s
    else:
        if Ne <= 0:
            raise ValueError("Ne must be positive")
        gamma = 2 * Ne * s

    return [
        2 * gamma + 2 * gamma,
        2 * gamma + 2 * h * gamma,
        2 * h * gamma + 2 * gamma,
        2 * h * gamma + 2 * h * gamma,
        2 * gamma,
        2 * h * gamma + 2 * h * gamma,
        2 * h * gamma,
        2 * gamma,
        2 * h * gamma,
    ]


def gene_based_dominance(s, h=0.5, Ne=None):
    """
    Returns the general selection parameters for the gene-based dominance model with
    a single selection and dominance coefficient that applies to haplotypes:
    [s_AB_AB, s_AB_Ab, s_AB_aB, s_AB_ab, s_Ab_Ab, s_Ab_aB, s_Ab_ab, s_aB_aB, s_aB_ab]
    """
    if Ne is None:
        gamma = s
    else:
        if Ne <= 0:
            raise ValueError("Ne must be positive")
        gamma = 2 * Ne * s

    return [
        2 * gamma,
        2 * gamma,
        2 * gamma,
        2 * h * gamma,
        2 * gamma,
        2 * gamma,
        2 * h * gamma,
        2 * gamma,
        2 * h * gamma,
    ]


##
## Methods for computing low-order statistics from the two-locus spectrum
##


def _compute_S(F_in, nA, nB):
    F = copy.deepcopy(F_in)
    F.mask_fixed()
    if nA == None and nB == None:
        return F.sum()
    else:
        if nA is not None:
            for ii in range(F.sample_size + 1):
                for jj in range(F.sample_size + 1 - ii):
                    if ii + jj != nA:
                        F.mask[ii, jj, :] = True
        if nB is not None:
            for ii in range(F.sample_size + 1):
                for kk in range(F.sample_size + 1 - ii):
                    if ii + kk != nB:
                        F.mask[ii, :, kk] = True
        return F.sum()


def _compute_D(F, proj, nA, nB):
    n = F.sample_size
    DD = 0
    if nA is None or nB is None:
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    if F.mask[ii, jj, kk] == True:
                        continue
                    if ii + jj == 0 or ii + kk == 0 or ii + jj == n or ii + kk == n:
                        continue
                    if nA is not None and nA != ii + jj:
                        continue
                    if nB is not None and nB != ii + kk:
                        continue
                    ll = n - ii - jj - kk
                    if proj is True:
                        DD += F.data[ii, jj, kk] * ((ii * ll - jj * kk) / (n * (n - 1)))
                    else:
                        DD += F.data[ii, jj, kk] * (ii * ll - jj * kk) / n ** 2
    else:
        nAB, ps = pAB(F, nA, nB)
        for ii, p in zip(nAB, ps):
            jj = nA - ii
            kk = nB - ii
            ll = n - ii - jj - kk
            assert jj >= 0 and kk >= 0 and ll >= 0
            if proj is True:
                DD += p * (ii * ll - jj * kk) / (n * (n - 1))
            else:
                DD += p * (ii * ll - jj * kk) / n ** 2
    return DD


def _compute_D2(F, proj, nA, nB):
    n = F.sample_size
    DD2 = 0
    if nA is None or nB is None:
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    if F.mask[ii, jj, kk] == True:
                        continue
                    if ii + jj == 0 or ii + kk == 0 or ii + jj == n or ii + kk == n:
                        continue
                    if nA is not None and nA != ii + jj:
                        continue
                    if nB is not None and nB != ii + kk:
                        continue
                    ll = n - ii - jj - kk
                    if proj == True:
                        DD2 += (
                            F.data[ii, jj, kk]
                            * (
                                ii * (ii - 1) * ll * (ll - 1)
                                + jj * (jj - 1) * kk * (kk - 1)
                                - 2 * ii * jj * kk * ll
                            )
                            / (n * (n - 1) * (n - 2) * (n - 3))
                        )
                    else:
                        DD2 += (
                            F.data[ii, jj, kk]
                            * (
                                ii ** 2 * ll ** 2
                                + jj ** 2 * kk ** 2
                                - 2 * ii * jj * kk * ll
                            )
                            / n ** 4
                        )
    else:
        nAB, ps = pAB(F, nA, nB)
        for ii, p in zip(nAB, ps):
            jj = nA - ii
            kk = nB - ii
            ll = n - ii - jj - kk
            assert jj >= 0 and kk >= 0 and ll >= 0
            if proj is True:
                DD2 += (
                    p
                    * (
                        ii * (ii - 1) * ll * (ll - 1)
                        + jj * (jj - 1) * kk * (kk - 1)
                        - 2 * ii * jj * kk * ll
                    )
                    / (n * (n - 1) * (n - 2) * (n - 3))
                )
            else:
                DD2 += (
                    p
                    * (ii ** 2 * ll ** 2 + jj ** 2 * kk ** 2 - 2 * ii * jj * kk * ll)
                    / n ** 4
                )

    return DD2


def _compute_Dz(F, proj, nA, nB):
    n = F.sample_size
    stat = 0
    if nA is None or nB is None:
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    if F.mask[ii, jj, kk] == True:
                        continue
                    if ii + jj == 0 or ii + kk == 0 or ii + jj == n or ii + kk == n:
                        continue
                    if nA is not None and nA != ii + jj:
                        continue
                    if nB is not None and nB != ii + kk:
                        continue
                    ll = n - ii - jj - kk
                    if proj == True:
                        stat += (
                            F.data[ii, jj, kk]
                            * (
                                -ii * (ii - 1) * jj * kk
                                + jj * (jj - 1) * (jj - 2) * kk
                                - 2 * jj * (jj - 1) * kk * (kk - 1)
                                + jj * kk * (kk - 1) * (kk - 2)
                                + ii * (ii - 1) * (ii - 2) * ll
                                - ii * jj * (jj - 1) * ll
                                + 4 * ii * jj * kk * ll
                                - ii * kk * (kk - 1) * ll
                                - 2 * ii * (ii - 1) * ll * (ll - 1)
                                - jj * kk * ll * (ll - 1)
                                + ii * ll * (ll - 1) * (ll - 2)
                            )
                            / (n * (n - 1) * (n - 2) * (n - 3))
                        )
                    else:
                        stat += (
                            F.data[ii, jj, kk]
                            * (
                                -(ii ** 2) * jj * kk
                                + jj ** 3 * kk
                                - 2 * jj ** 2 * kk ** 2
                                + jj * kk ** 3
                                + ii ** 3 * ll
                                - ii * jj ** 2 * ll
                                + 4 * ii * jj * kk * ll
                                - ii * kk ** 2 * ll
                                - 2 * ii ** 2 * ll ** 2
                                - jj * kk * ll ** 2
                                + ii * ll ** 3
                            )
                            / n ** 4
                        )
    else:
        nAB, ps = pAB(F, nA, nB)
        for ii, p in zip(nAB, ps):
            jj = nA - ii
            kk = nB - ii
            ll = n - ii - jj - kk
            assert jj >= 0 and kk >= 0 and ll >= 0
            if proj is True:
                stat += (
                    p
                    * (
                        -ii * (ii - 1) * jj * kk
                        + jj * (jj - 1) * (jj - 2) * kk
                        - 2 * jj * (jj - 1) * kk * (kk - 1)
                        + jj * kk * (kk - 1) * (kk - 2)
                        + ii * (ii - 1) * (ii - 2) * ll
                        - ii * jj * (jj - 1) * ll
                        + 4 * ii * jj * kk * ll
                        - ii * kk * (kk - 1) * ll
                        - 2 * ii * (ii - 1) * ll * (ll - 1)
                        - jj * kk * ll * (ll - 1)
                        + ii * ll * (ll - 1) * (ll - 2)
                    )
                    / (n * (n - 1) * (n - 2) * (n - 3))
                )
            else:
                stat += (
                    p
                    * (
                        -(ii ** 2) * jj * kk
                        + jj ** 3 * kk
                        - 2 * jj ** 2 * kk ** 2
                        + jj * kk ** 3
                        + ii ** 3 * ll
                        - ii * jj ** 2 * ll
                        + 4 * ii * jj * kk * ll
                        - ii * kk ** 2 * ll
                        - 2 * ii ** 2 * ll ** 2
                        - jj * kk * ll ** 2
                        + ii * ll ** 3
                    )
                    / n ** 4
                )

    return stat


def _compute_pi2(F, proj, nA, nB):
    n = F.sample_size
    stat = 0
    if nA is None or nB is None:
        for ii in range(n + 1):
            for jj in range(n + 1 - ii):
                for kk in range(n + 1 - ii - jj):
                    if F.mask[ii, jj, kk] == True:
                        continue
                    ll = n - ii - jj - kk
                    if ii + jj == 0 or ii + kk == 0 or ii + jj == n or ii + kk == n:
                        continue
                    if nA is not None and nA != ii + jj:
                        continue
                    if nB is not None and nB != ii + kk:
                        continue
                    if proj == True:
                        stat += (
                            F.data[ii, jj, kk]
                            * (
                                ii * (ii - 1) * jj * kk
                                + ii * jj * (jj - 1) * kk
                                + ii * jj * kk * (kk - 1)
                                + jj * (jj - 1) * kk * (kk - 1)
                                + ii * (ii - 1) * jj * ll
                                + ii * jj * (jj - 1) * ll
                                + ii * (ii - 1) * kk * ll
                                + 2 * ii * jj * kk * ll
                                + jj * (jj - 1) * kk * ll
                                + ii * kk * (kk - 1) * ll
                                + jj * kk * (kk - 1) * ll
                                + ii * (ii - 1) * ll * (ll - 1)
                                + ii * jj * ll * (ll - 1)
                                + ii * kk * ll * (ll - 1)
                                + jj * kk * ll * (ll - 1)
                            )
                            / (n * (n - 1) * (n - 2) * (n - 3))
                        )
                    else:
                        stat += (
                            F.data[ii, jj, kk]
                            * (
                                ii ** 2 * jj * kk
                                + ii * jj ** 2 * kk
                                + ii * jj * kk ** 2
                                + jj ** 2 * kk ** 2
                                + ii ** 2 * jj * ll
                                + ii * jj ** 2 * ll
                                + ii ** 2 * kk * ll
                                + 2 * ii * jj * kk * ll
                                + jj ** 2 * kk * ll
                                + ii * kk ** 2 * ll
                                + jj * kk ** 2 * ll
                                + ii ** 2 * ll ** 2
                                + ii * jj * ll ** 2
                                + ii * kk * ll ** 2
                                + jj * kk * ll ** 2
                            )
                            / n ** 4
                        )
    else:
        nAB, ps = pAB(F, nA, nB)
        for ii, p in zip(nAB, ps):
            jj = nA - ii
            kk = nB - ii
            ll = n - ii - jj - kk
            assert jj >= 0 and kk >= 0 and ll >= 0
            if proj is True:
                stat += (
                    p
                    * (
                        ii * (ii - 1) * jj * kk
                        + ii * jj * (jj - 1) * kk
                        + ii * jj * kk * (kk - 1)
                        + jj * (jj - 1) * kk * (kk - 1)
                        + ii * (ii - 1) * jj * ll
                        + ii * jj * (jj - 1) * ll
                        + ii * (ii - 1) * kk * ll
                        + 2 * ii * jj * kk * ll
                        + jj * (jj - 1) * kk * ll
                        + ii * kk * (kk - 1) * ll
                        + jj * kk * (kk - 1) * ll
                        + ii * (ii - 1) * ll * (ll - 1)
                        + ii * jj * ll * (ll - 1)
                        + ii * kk * ll * (ll - 1)
                        + jj * kk * ll * (ll - 1)
                    )
                    / (n * (n - 1) * (n - 2) * (n - 3))
                )
            else:
                stat += (
                    p
                    * (
                        ii ** 2 * jj * kk
                        + ii * jj ** 2 * kk
                        + ii * jj * kk ** 2
                        + jj ** 2 * kk ** 2
                        + ii ** 2 * jj * ll
                        + ii * jj ** 2 * ll
                        + ii ** 2 * kk * ll
                        + 2 * ii * jj * kk * ll
                        + jj ** 2 * kk * ll
                        + ii * kk ** 2 * ll
                        + jj * kk ** 2 * ll
                        + ii ** 2 * ll ** 2
                        + ii * jj * ll ** 2
                        + ii * kk * ll ** 2
                        + jj * kk * ll ** 2
                    )
                    / n ** 4
                )

    return stat


##
## The above methods compute statistics for given nA and nB slices, if provided.
## We may also want to compute statistics with a given allele count threshold.
## To do so,
##


def compute_D_threshold(F, proj=True, thresh=None):
    """
    Given a frequency spectrum, compute D over all frequencies
    with nA and nB <= thresh.
    """
    if thresh is None:
        return _compute_D(F, proj, None, None)
    else:
        stat = 0
        for nA in range(1, thresh + 1):
            for nB in range(1, thresh + 1):
                stat += _compute_D(F, proj, nA, nB)
        return stat


def compute_D2_threshold(F, proj=True, thresh=None):
    """
    Given a frequency spectrum, compute D2 over all frequencies
    with nA and nB <= thresh.
    """
    if thresh is None:
        return _compute_D2(F, proj, None, None)
    else:
        stat = 0
        for nA in range(1, thresh + 1):
            for nB in range(1, thresh + 1):
                stat += _compute_D2(F, proj, nA, nB)
        return stat


def compute_Dz_threshold(F, proj=True, thresh=None):
    """
    Given a frequency spectrum, compute Dz over all frequencies
    with nA and nB <= thresh.
    """
    if thresh is None:
        return _compute_Dz(F, proj, None, None)
    else:
        stat = 0
        for nA in range(1, thresh + 1):
            for nB in range(1, thresh + 1):
                stat += _compute_Dz(F, proj, nA, nB)
        return stat


def compute_pi2_threshold(F, proj=True, thresh=None):
    """
    Given a frequency spectrum, compute pi2 over all frequencies
    with nA and nB <= thresh.
    """
    if thresh is None:
        return _compute_pi2(F, proj, None, None)
    else:
        stat = 0
        for nA in range(1, thresh + 1):
            for nB in range(1, thresh + 1):
                stat += _compute_pi2(F, proj, nA, nB)
        return stat


def compute_D_conditional(F, proj=True, nAmin=None, nAmax=None, nBmin=None, nBmax=None):
    """
    Given a frequency spectrum, compute D over all frequencies
    given by the conditions.
    If a condition is None, that condition is not constrained.
    """
    if nAmin is None:
        nAmin = 1
    if nAmax is None:
        nAmax = F.sample_size - 1
    if nBmin is None:
        nBmin = 1
    if nBmax is None:
        nBmax = F.sample_size - 1
    stat = 0
    for nA in range(nAmin, nAmax + 1):
        for nB in range(nBmin, nBmax + 1):
            stat += _compute_D(F, proj, nA, nB)
    return stat


def compute_D2_conditional(
    F, proj=True, nAmin=None, nAmax=None, nBmin=None, nBmax=None
):
    """
    Given a frequency spectrum, compute D2 over all frequencies
    given by the conditions.
    If a condition is None, that condition is not constrained.
    """
    if nAmin is None:
        nAmin = 1
    if nAmax is None:
        nAmax = F.sample_size - 1
    if nBmin is None:
        nBmin = 1
    if nBmax is None:
        nBmax = F.sample_size - 1
    stat = 0
    for nA in range(nAmin, nAmax + 1):
        for nB in range(nBmin, nBmax + 1):
            stat += _compute_D2(F, proj, nA, nB)
    return stat


def compute_Dz_conditional(
    F, proj=True, nAmin=None, nAmax=None, nBmin=None, nBmax=None
):
    """
    Given a frequency spectrum, compute Dz over all frequencies
    given by the conditions.
    If a condition is None, that condition is not constrained.
    """
    if nAmin is None:
        nAmin = 1
    if nAmax is None:
        nAmax = F.sample_size - 1
    if nBmin is None:
        nBmin = 1
    if nBmax is None:
        nBmax = F.sample_size - 1
    stat = 0
    for nA in range(nAmin, nAmax + 1):
        for nB in range(nBmin, nBmax + 1):
            stat += _compute_Dz(F, proj, nA, nB)
    return stat


def compute_pi2_conditional(
    F, proj=True, nAmin=None, nAmax=None, nBmin=None, nBmax=None
):
    """
    Given a frequency spectrum, compute pi2 over all frequencies
    given by the conditions.
    If a condition is None, that condition is not constrained.
    """
    if nAmin is None:
        nAmin = 1
    if nAmax is None:
        nAmax = F.sample_size - 1
    if nBmin is None:
        nBmin = 1
    if nBmax is None:
        nBmax = F.sample_size - 1
    stat = 0
    for nA in range(nAmin, nAmax + 1):
        for nB in range(nBmin, nBmax + 1):
            stat += _compute_pi2(F, proj, nA, nB)
    return stat
