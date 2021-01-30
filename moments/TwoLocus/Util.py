import numpy as np


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

    gamma *= 2

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
