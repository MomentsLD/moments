import numpy as np
from moments.LD.LDstats_mod import LDstats
from . import Numerics


def snm(params=None, order=2, rho=None, theta=0.001, pop_ids=None):
    """
    Equilibrium neutral model. Does not take demographic parameters.

    :param params: Unused.
    :param order: The maximum order of the LD statistics. Defaults to 2.
    :type order: int
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    Y = Numerics.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    return Y


def two_epoch(params, order=2, rho=None, theta=0.001, pop_ids=None):
    """
    Two epoch model with a single size change and constant sized epochs.

    :param params: The relative size and integration time of recent epoch,
        in genetic units: (nu, T).
    :type params: list
    :param order: The maximum order of the LD statistics. Defaults to 2.
    :type order: int
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nu, T = params
    Y = Numerics.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    Y.integrate([nu], T, rho=rho, theta=theta)
    return Y


def three_epoch(params, order=2, rho=None, theta=0.001, pop_ids=None):
    """
    Three epoch model with constant sized epochs.

    :param params: The relative sizes and integration times of recent epochs,
        in genetic units: (nu1, nu2, T1, T2).
    :type params: list
    :param order: The maximum order of the LD statistics. Defaults to 2.
    :type order: int
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nu1, nu2, T1, T2 = params
    Y = Numerics.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    Y.integrate([nu1], T1, rho=rho, theta=theta)
    Y.integrate([nu2], T2, rho=rho, theta=theta)
    return Y


def growth(params, order=2, rho=None, theta=0.001, pop_ids=None):
    """
    Exponential growth (or decay) model.

    :param params: The relative final size and integration time of recent epoch,
        in genetic units: (nuF, T)
    :type params: list
    :param order: The maximum order of the LD statistics. Defaults to 2.
    :type order: int
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nuF, T = params
    Y = Numerics.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    nu_func = lambda t: [np.exp(np.log(nuF) * t / T)]
    Y.integrate(nu_func, T, rho=rho, theta=theta)
    return Y


def bottlegrowth(params, order=2, rho=None, theta=0.001, pop_ids=None):
    """
    Exponential growth (or decay) model after size change.

    :param params: The relative initial and final sizes of the final epoch
        and its integration time in genetic units: (nuB, nuF, T).
    :type params: list
    :param order: The maximum order of the LD statistics. Defaults to 2.
    :type order: int
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nuB, nuF, T = params
    Y = Numerics.steady_state([1], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1, pop_ids=pop_ids)
    nu_func = lambda t: [nuB * np.exp(np.log(nuF / nuB) * t / T)]
    Y.integrate(nu_func, T, rho=rho, theta=theta)
    return Y
