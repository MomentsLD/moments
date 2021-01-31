import numpy as np


def pAB(F, nA, nB):
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
