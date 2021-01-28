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
