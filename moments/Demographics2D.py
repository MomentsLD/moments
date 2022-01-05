"""
Two-population demographic models.
"""
import numpy

import moments


def snm(ns, pop_ids=None):
    """
    ns = [n1, n2]

    Standard neutral model with a split but no divergence.

    :param ns: List of population sizes in first and second populations.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.pop_ids = pop_ids
    return fs


def bottlegrowth(params, ns, pop_ids=None):
    """
    params = (nuB, nuF, T)

    ns = [n1, n2]

    Instantanous size change followed by exponential growth with no population
    split.

    - nuB: Ratio of population size after instantanous change to ancient
      population size
    - nuF: Ratio of contempoary to ancient population size
    - T: Time in the past at which instantaneous change happened and growth began
      (in units of 2*Na generations)
    - n1, n2: Sample sizes of resulting Spectrum.

    :param params: List of parameters, (nuB, nuF, T).
    :param ns: List of population sizes in first and second populations.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    nuB, nuF, T = params
    return bottlegrowth_split_mig((nuB, nuF, 0, T, 0), ns, pop_ids=pop_ids)


def bottlegrowth_split(params, ns, pop_ids=None):
    """
    params = (nuB, nuF, T, Ts)

    ns = [n1, n2]

    Instantanous size change followed by exponential growth then split.

    - nuB: Ratio of population size after instantanous change to ancient
      population size
    - nuF: Ratio of contempoary to ancient population size
    - T: Time in the past at which instantaneous change happened and growth began
      (in units of 2*Na generations)
    - Ts: Time in the past at which the two populations split.
    - n1, n2: Sample sizes of resulting Spectrum.

    :param ns: List of population sizes in first and second populations.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    nuB, nuF, T, Ts = params
    return bottlegrowth_split_mig((nuB, nuF, 0.0, T, Ts), ns, pop_ids=pop_ids)


def bottlegrowth_split_mig(params, ns, pop_ids=None):
    """
    params = (nuB, nuF, m, T, Ts)
    ns = [n1, n2]

    Instantanous size change followed by exponential growth then split with
    migration.

    - nuB: Ratio of population size after instantanous change to ancient
      population size
    - nuF: Ratio of contempoary to ancient population size
    - m: Migration rate between the two populations (2*Na*m).
    - T: Time in the past at which instantaneous change happened and growth began
      (in units of 2*Na generations)
    - Ts: Time in the past at which the two populations split.
    - n1, n2: Sample sizes of resulting Spectrum.

    :param ns: List of population sizes in first and second populations.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    nuB, nuF, m, T, Ts = params
    nu_func = lambda t: [nuB * numpy.exp(numpy.log(nuF / nuB) * t / T)]
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs.integrate(nu_func, T - Ts, dt_fac=0.01)
    # we split the population
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nu0 = nu_func(T - Ts)[0]
    nu_func = lambda t: 2 * [nu0 * numpy.exp(numpy.log(nuF / nu0) * t / Ts)]
    fs.integrate(nu_func, Ts, m=numpy.array([[0, m], [m, 0]]))
    fs.pop_ids = pop_ids
    return fs


def split_mig(params, ns, pop_ids=None):
    """
    Split into two populations of specifed size, with migration.

    params = (nu1, nu2, T, m)

    ns = [n1, n2]

    :param params: Tuple of length 4.

        - nu1: Size of population 1 after split.
        - nu2: Size of population 2 after split.
        - T: Time in the past of split (in units of 2*Na generations)
        - m: Migration rate between populations (2*Na*m)
    :param ns: List of length two specifying sample sizes n1 and n2.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    nu1, nu2, T, m = params
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1, nu2], T, m=numpy.array([[0, m], [m, 0]]))
    fs.pop_ids = pop_ids
    return fs


def IM(params, ns, pop_ids=None):
    """
    Isolation-with-migration model with exponential pop growth.

    params = (s, nu1, nu2, T, m12, m21)

    ns = [n1, n2]

    :param params: Tuple of length 6.

        - s: Size of pop 1 after split. (Pop 2 has size 1-s.)
        - nu1: Final size of pop 1.
        - nu2: Final size of pop 2.
        - T: Time in the past of split (in units of 2*Na generations)
        - m12: Migration from pop 2 to pop 1 (2 * Na * m12)
        - m21: Migration from pop 1 to pop 2
    :param ns: List of population sizes in first and second populations.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    s, nu1, nu2, T, m12, m21 = params
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nu1_func = lambda t: s * (nu1 / s) ** (t / T)
    nu2_func = lambda t: (1 - s) * (nu2 / (1 - s)) ** (t / T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
    fs.integrate(nu_func, T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))
    fs.pop_ids = pop_ids
    return fs


def IM_pre(params, ns, pop_ids=None):
    """
    params = (nuPre, TPre, s, nu1, nu2, T, m12, m21)

    ns = [n1, n2]

    Isolation-with-migration model with exponential pop growth and a size change
    prior to split.

    - nuPre: Size after first size change
    - TPre: Time before split of first size change.
    - s: Fraction of nuPre that goes to pop1. (Pop 2 has size nuPre*(1-s).)
    - nu1: Final size of pop 1.
    - nu2: Final size of pop 2.
    - T: Time in the past of split (in units of 2*Na generations)
    - m12: Migration from pop 2 to pop 1 (2*Na*m12)
    - m21: Migration from pop 1 to pop 2
    - n1, n2: Sample sizes of resulting Spectrum.

    :param ns: List of population sizes in first and second populations.
    :param pop_ids: List of population IDs.
    """
    if pop_ids is not None and len(pop_ids) != 2:
        raise ValueError("pop_ids must be a list of two population IDs")
    nuPre, TPre, s, nu1, nu2, T, m12, m21 = params
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs.integrate([nuPre], TPre, dt_fac=0.01)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nu1_0 = nuPre * s
    nu2_0 = nuPre * (1 - s)
    nu1_func = lambda t: nu1_0 * (nu1 / nu1_0) ** (t / T)
    nu2_func = lambda t: nu2_0 * (nu2 / nu2_0) ** (t / T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
    fs.integrate(nu_func, T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))
    fs.pop_ids = pop_ids
    return fs
