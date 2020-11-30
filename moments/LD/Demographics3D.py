"""
Three-population demographic models.
"""
import numpy as np
import moments.LD


def out_of_Africa(params, rho=None, theta=0.001, pop_ids=["YRI", "CEU", "CHB"]):
    """
    The Gutenkunst et al (2009) out-of-Africa that has been reinferred a
    number of times.

    :param params: List of parameters, in the order (nuA, TA, nuB, TB, nuEu0,
        nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs).
    :param rho: Recombination rate or list of recombination rates (population-size
        scaled).
    :param theta: Population-size scaled mutation rate.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 3:
        raise ValueError("pop_ids must be a list of three population IDs")
    (
        nuA,
        TA,
        nuB,
        TB,
        nuEu0,
        nuEuF,
        nuAs0,
        nuAsF,
        TF,
        mAfB,
        mAfEu,
        mAfAs,
        mEuAs,
    ) = params
    y = moments.LD.Demographics1D.snm(rho=rho, theta=theta)
    # integrate modern human branch with expansion
    y.integrate([nuA], TA, rho=rho, theta=theta)
    # split into African and Eurasian branches and integrate
    y = y.split(0)
    mig_mat = [[0, mAfB], [mAfB, 0]]
    y.integrate([nuA, nuB], TB, m=mig_mat, rho=rho, theta=theta)
    # split Eurasian into CEU and CHB
    y = y.split(1)
    nu_func = lambda t: [
        nuA,
        nuEu0 * np.exp(np.log(nuEuF / nuEu0) * t / TF),
        nuAs0 * np.exp(np.log(nuAsF / nuAs0) * t / TF),
    ]
    mig_mat = [[0, mAfEu, mAfAs], [mAfEu, 0, mEuAs], [mAfAs, mEuAs, 0]]
    y.integrate(nu_func, TF, m=mig_mat, rho=rho, theta=theta)
    y.pop_ids = pop_ids
    return y
