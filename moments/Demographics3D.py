"""
Three-population demographic models.
"""
import numpy as np
import moments


def out_of_Africa(params, ns, pop_ids=["YRI", "CEU", "CHB"]):
    """
    The Gutenkunst et al (2009) out-of-Africa that has been reinferred a
    number of times.

    :param params: List of parameters, in the order (nuA, TA, nuB, TB, nuEu0,
        nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs, mEuAs).
    :type params: list of floats
    :param ns: List of population sizes in each population, in order given
        by `pop_ids`.
    :type ns: list of ints
    :param pop_ids: List of population IDs, defaults to ["YRI", "CEU", "CHB"].
    :type pop_ids: list of strings, optional
    """
    if pop_ids is not None and len(pop_ids) != 3:
        raise ValueError("pop_ids must be a list of three population IDs")
    if len(ns) != 3:
        raise ValueError("ns must have length 3")
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
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1] + ns[2])
    fs = moments.Spectrum(sts)
    # integrate modern human branch with expansion
    fs.integrate([nuA], TA)
    # split into African and Eurasian branches and integrate
    fs = fs.split(0, ns[0], ns[1] + ns[2])
    mig_mat = [[0, mAfB], [mAfB, 0]]
    fs.integrate([nuA, nuB], TB, m=mig_mat)
    # split Eurasian into CEU and CHB
    fs = fs.split(1, ns[1], ns[2])
    nu_func = lambda t: [
        nuA,
        nuEu0 * np.exp(np.log(nuEuF / nuEu0) * t / TF),
        nuAs0 * np.exp(np.log(nuAsF / nuAs0) * t / TF),
    ]
    mig_mat = [[0, mAfEu, mAfAs], [mAfEu, 0, mEuAs], [mAfAs, mEuAs, 0]]
    fs.integrate(nu_func, TF, m=mig_mat)
    fs.pop_ids = pop_ids
    return fs
