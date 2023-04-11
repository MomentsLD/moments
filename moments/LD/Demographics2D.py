import numpy as np
from moments.LD.LDstats_mod import LDstats
from . import Numerics


def snm(params=None, rho=None, theta=0.001, pop_ids=None):
    """
    Equilibrium neutral model. Neutral steady state followed by split in
    the immediate past.

    :param params: Unused.
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 2.
    :type pop_ids: lits of str, optional
    """
    Y = Numerics.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1)
    Y = Y.split(0)
    Y.pop_ids = pop_ids
    return Y


def split_mig(params, rho=None, theta=0.001, pop_ids=None):
    """
    Split into two populations of specifed size, which then have their own
    relative constant sizes and symmetric migration between populations.

    - nu1: Size of population 1 after split.
    - nu2: Size of population 2 after split.
    - T: Time in the past of split (in units of 2*Na generations)
    - m: Migration rate between populations (2*Na*m)

    :param params: The input parameters: (nu1, nu2, T, m)
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nu1, nu2, T, m = params

    Y = snm(rho=rho, theta=theta)
    Y.integrate([nu1, nu2], T, rho=rho, theta=theta, m=[[0, m], [m, 0]])
    Y.pop_ids = pop_ids
    return Y


def island_model(params, rho=None, theta=0.001, pop_ids=None):
    """
    Split into two populations of specifed size, which then have their own
    relative constant sizes and symmetric migration between populations.

    - nu1: Relative size of population 1 after split.
    - nu2: Relative size of population 2 after split.
    - m12: Migration rate, from 2 to 1 forward in time (in units 2*Ne*m)
    - m21: Migration rate, from 1 to 2 forward in time (in units 2*Ne*m)

    :param params: The input parameters: (nu1, nu2, m12, m21)
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nu1, nu2, m12, m21 = params

    Y = Numerics.steady_state([nu1, nu2], m=[[0, m12], [m21, 0]], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=2, pop_ids=pop_ids)
    return Y
