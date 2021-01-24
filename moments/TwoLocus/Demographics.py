import numpy as np
import moments.TwoLocus.Integration
import moments.TwoLocus
import os

"""
Demographic models for two locus model (all single population)
Two-epoch
Three-epoch
Growth
Bottlegrowth
"""

# Cache equilibrium spectra in ~/.moments/TwoLocus_cache by default
def set_cache_path(path="~/.moments/TwoLocus_cache"):
    """
    Set directory in which demographic equilibrium phi spectra will be cached.

    The collection of cached spectra can get large, so it may be helpful to
    store them outside the user's home directory.
    """
    global cache_path
    cache_path = os.path.expanduser(path)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)


cache_path = None
set_cache_path()


def _make_floats(params):
    """
    pass list of params, return floats of those params (for caching)
    """
    if params is None:
        return None
    if hasattr(params, "__len__"):
        return [float(p) for p in params]
    else:
        return float(params)


def equilibrium(ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    Compute or load the equilibrium two locus frequency spectrum. If the cached spectrum
    does not exist, create the equilibrium spectrum and cache in the cache path.

    :param ns: The sample size.
    :param rho: The population size scaled selection coefficient, 4*Ne*r.
    :param theta: The mutation rate at each locus, typically left as 1.
    :param gamma: Only used for additive selection at the A/a locus.
    :param sel_params: Additive selection coefficients for haplotypes AB, Ab, and aB, so
        that sel_params = [sAB, sA, sB]. If sAB = sA + sB, this is a model with no
        epistasis.
    """
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0

    if gamma == None:
        gamma = 0.0

    gamma = _make_floats(gamma)
    rho = _make_floats(rho)
    theta = _make_floats(theta)
    sel_params = _make_floats(sel_params)

    # fetch from cache if neutral (cache only neutral spectra for the moment)
    if gamma == 0.0 and (
        sel_params == None
        or (sel_params[0] == 0.0 and sel_params[1] == 0.0 and sel_params[2] == 0.0)
    ):
        eq_name = f"tlfs.ns_{ns}.rho_{rho}.theta_{theta}.fs"
        eq_name = os.path.join(cache_path, eq_name)
    elif gamma != 0.0:
        eq_name = f"tlfs.ns_{ns}.rho_{rho}.theta_{theta}.gammaA_{gamma}.fs"
        eq_name = os.path.join(cache_path, eq_name)
    elif sel_params != None:
        eq_name = f"tlfs.ns_{ns}.rho_{rho}.theta_{theta}.sel_{sel_params[0]}_{sel_params[1]}_{sel_params[2]}.fs"
        eq_name = os.path.join(cache_path, eq_name)

    try:
        F = moments.TwoLocus.TLSpectrum.from_file(eq_name)
    except IOError:
        F = np.zeros((ns + 1, ns + 1, ns + 1))
        ### I would rather compute the equilibrium state by inverting the transition matrix
        ### so long as it is not poory conditioned (note 7/16)
        F = moments.TwoLocus.Integration.integrate(
            F,
            1.0,
            40.0,
            rho=rho,
            theta=theta,
            gamma=gamma,
            sel_params=sel_params,
            dt=0.001,
        )
        F = moments.TwoLocus.TLSpectrum(F)
        F.to_file(eq_name)
    return F


def two_epoch(params, ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    A two-epoch model, with relative size change nu, time T in the past. T is given
    in units of 2Ne generations. Note that a relative size of 1 implies no size change.

    :param params: Given as [nu, T].
    :param ns: The sample size.
    :param rho: The population size scaled selection coefficient, 4*Ne*r.
    :param theta: The mutation rate at each locus, typically left as 1.
    :param gamma: Only used for additive selection at the A/a locus.
    :param sel_params: Additive selection coefficients for haplotypes AB, Ab, and aB, so
        that sel_params = [sAB, sA, sB]. If sAB = sA + sB, this is a model with no
        epistasis.
    """
    nu, T = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0

    if gamma == None:
        gamma = 0.0

    gamma = _make_floats(gamma)
    rho = _make_floats(rho)
    theta = _make_floats(theta)
    sel_params = _make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    F.integrate(nu, T, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    return F


def three_epoch(params, ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    A three-epoch model, with relative size changes nu1 that lasts for time T1, followed
    by a relative size change to nu2 that last for time T2. Times are in units of 2Ne
    generations, and sizes are relative to the ancestral Ne.

    :param params: Given as [nu1, nu2, T1, T2].
    :param ns: The sample size.
    :param rho: The population size scaled selection coefficient, 4*Ne*r.
    :param theta: The mutation rate at each locus, typically left as 1.
    :param gamma: Only used for additive selection at the A/a locus.
    :param sel_params: Additive selection coefficients for haplotypes AB, Ab, and aB, so
        that sel_params = [sAB, sA, sB]. If sAB = sA + sB, this is a model with no
        epistasis.
    """
    nu1, nu2, T1, T2 = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0

    if gamma == None:
        gamma = 0.0

    gamma = _make_floats(gamma)
    rho = _make_floats(rho)
    theta = _make_floats(theta)
    sel_params = _make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    F.integrate(nu1, T1, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    F.integrate(nu2, T2, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    return F


def growth(params, ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    An expnential growth model, that begins growth at time T ago, in units of 2Ne
    generations. The final size is given by nu, which is the relative size to the
    ancestral Ne.

    :param params: Given as [nu, T].
    :param ns: The sample size.
    :param rho: The population size scaled selection coefficient, 4*Ne*r.
    :param theta: The mutation rate at each locus, typically left as 1.
    :param gamma: Only used for additive selection at the A/a locus.
    :param sel_params: Additive selection coefficients for haplotypes AB, Ab, and aB, so
        that sel_params = [sAB, sA, sB]. If sAB = sA + sB, this is a model with no
        epistasis.
    """
    nu, T = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0

    if gamma == None:
        gamma = 0.0

    gamma = _make_floats(gamma)
    rho = _make_floats(rho)
    theta = _make_floats(theta)
    sel_params = _make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    nu_func = lambda t: np.exp(np.log(nu) * t / T)
    F.integrate(
        nu_func, T, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params
    )
    return F


def bottlegrowth(params, ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    A bottleneck followed by exponential growth. The population changes size to nuB
    T generations ago, and then has exponential size change to final size nuF. Time is
    in units of 2Ne generations, and sizes are relative to the ancestral Ne.

    :param params: Given as [nuB, nuF, T].
    :param ns: The sample size.
    :param rho: The population size scaled selection coefficient, 4*Ne*r.
    :param theta: The mutation rate at each locus, typically left as 1.
    :param gamma: Only used for additive selection at the A/a locus.
    :param sel_params: Additive selection coefficients for haplotypes AB, Ab, and aB, so
        that sel_params = [sAB, sA, sB]. If sAB = sA + sB, this is a model with no
        epistasis.
    """
    nuB, nuF, T = params
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0

    if gamma == None:
        gamma = 0.0

    gamma = _make_floats(gamma)
    rho = _make_floats(rho)
    theta = _make_floats(theta)
    sel_params = _make_floats(sel_params)

    F = equilibrium(ns, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    nu_func = lambda t: nuB * np.exp(np.log(nuF / nuB) * t / T)
    F.integrate(
        nu_func, T, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params
    )
    return F
