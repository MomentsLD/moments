"""
Single population demographic models.
"""
import numpy

import moments


def snm(ns):
    """
    Standard neutral model with theta=1.

    :param ns: Number of samples in resulting Spectrum. Must be a list
        of length one.
    """
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    return fs


def two_epoch(params, ns):
    """
    Instantaneous size change some time ago.

    params = (nu,T)
    
    :param params: Tuple of length two, specifying (nu, T).
        
        nu: the ratio of contemporary to ancient population size.
        
        T: the time in the past at which size change happened
        (in units of 2*Ne generations).
    :param ns: Number of samples in resulting Spectrum. Must be a list of
        length one.
    """
    nu, T = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    fs.integrate([nu], T)
    return fs


def growth(params, ns):
    """
    Exponential growth beginning some time ago.

    params = (nu,T)

    :param params: Tupe of length two, specifying (nu, t).
        
        nu: the final population size.
        
        T: the time in the past at which growth began
        (in units of 2*Ne generations).
    :param ns: Number of samples in resulting Spectrum. Must be a list of
        length one.
    """
    nu, T = params

    nu_func = lambda t: [numpy.exp(numpy.log(nu) * t / T)]
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    fs.integrate(nu_func, T, 0.01)

    return fs


def bottlegrowth(params, ns):
    """
    Instantanous size change followed by exponential growth.

    params = (nuB,nuF,T)

    :param params: Tuple of length three specifying (nuB, nuF, T).
        
        nuB: Ratio of population size after instantanous change to ancient
        population size.
        
        nuF: Ratio of contemporary to ancient population size.
        
        T: Time in the past at which instantaneous change happened and growth began
        (in units of 2*Na generations).
    :param ns: Number of samples in resulting Spectrum.
    """
    nuB, nuF, T = params
    nu_func = lambda t: [nuB * numpy.exp(numpy.log(nuF / nuB) * t / T)]

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    fs.integrate(nu_func, T, 0.01)

    return fs


def three_epoch(params, ns):
    """
    Three epoch model of constant sizes.
    
    params = (nuB, nuF, TB, TF)

    :param params: Tuple of length four specifying (nuB, nuF, TB, TF).
        
        nuB: Ratio of bottleneck population size to ancient pop size.
        
        nuF: Ratio of contemporary to ancient pop size.
        
        TB: Length of bottleneck (in units of 2*Na generations).
        
        TF: Time since bottleneck recovery (in units of 2*Na generations).
    :param ns: Number of samples in resulting Spectrum.
    """
    nuB, nuF, TB, TF = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0])
    fs = moments.Spectrum(sts)
    fs.integrate([nuB], TB, 0.01)
    fs.integrate([nuF], TF, 0.01)

    return fs
