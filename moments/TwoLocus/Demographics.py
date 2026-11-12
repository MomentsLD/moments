"""
Equilibrium spectra and simple single-population demographic models for the two-locus
frequency spectrum: two-epoch, three-epoch, exponential growth, and bottleneck followed
by exponential growth.

All models use the infinite sites mutation model. Selection may be specified with
``gamma`` (additive selection with the same coefficient at both loci and no epistasis)
or with ``sel_params`` (additive selection within loci, allowing for different
coefficients at the two loci and epistasis). :func:`equilibrium` also accepts
``sel_params_general``, which allows for dominance.
"""

import numpy as np
import moments.TwoLocus.Integration
import moments.TwoLocus
import os


# Cache equilibrium spectra in ~/.moments/TwoLocus_cache by default
def set_cache_path(path="~/.moments/TwoLocus_cache"):
    """
    Set directory in which equilibrium two-locus spectra will be cached, when
    :func:`equilibrium` is called with ``cache=True``. Defaults to
    ``~/.moments/TwoLocus_cache``.

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


def _equilibrium_cache_name(ns, rho, theta, sel_params, sel_params_general):
    """
    Path of the cached equilibrium spectrum. Every input that changes the spectrum must
    appear in the file name, so that different models never share a cache entry.
    """
    eq_name = f"tlfs.ns_{ns}.rho_{rho}.theta_{theta}"
    if sel_params is not None:
        eq_name += ".sel_" + "_".join(str(s) for s in sel_params)
    if sel_params_general is not None:
        eq_name += ".sel_general_" + "_".join(str(s) for s in sel_params_general)
    return os.path.join(cache_path, eq_name + ".fs")


def equilibrium(
    ns,
    rho=None,
    theta=1.0,
    gamma=None,
    sel_params=None,
    sel_params_general=None,
    cache=False,
):
    """
    Compute (or load from the cache) the steady-state two-locus frequency spectrum
    under the infinite sites model, for a population of constant size.

    :param ns: The haploid sample size.
    :param rho: The population-size scaled recombination rate between the two loci,
        4*Ne*r. If not given, rho = 0 is used and a warning is printed.
    :param theta: The population-size scaled mutation rate at each locus, 4*Ne*u.
        Defaults to 1.
    :param gamma: The population-size scaled selection coefficient, 2*Ne*s, for
        additive selection with the same coefficient at both loci and no epistasis.
        This is shorthand for ``sel_params = [2 * gamma, gamma, gamma]``, so that the
        single-locus spectrum at each locus matches the single-site SFS with selection
        coefficient gamma and h = 1/2. For selection at only one locus, unequal
        selection, or epistasis, use ``sel_params`` instead. Cannot be combined with
        ``sel_params``.
    :param sel_params: Additive (within-locus) selection parameters for the AB, Ab, and
        aB haplotypes relative to ab, given as ``[gamma_AB, gamma_A, gamma_B]``, where
        each is 2*Ne*s for that haplotype. For example, selection only at the left locus
        is ``[gamma, gamma, 0]``. With ``gamma_AB = gamma_A + gamma_B`` there is no
        epistasis. See :func:`moments.TwoLocus.Util.additive_epistasis`.
    :param sel_params_general: Selection parameters for the nine diploid two-locus
        genotypes relative to ab/ab, each given as 2*Ne*s for that genotype, in the order
        ``[AB/AB, AB/Ab, AB/aB, AB/ab, Ab/Ab, Ab/aB, Ab/ab, aB/aB, aB/ab]``. This model
        allows for dominance and epistasis, and requires ``ns >= 13``. Cannot be
        combined with ``gamma`` or ``sel_params``. See
        :func:`moments.TwoLocus.Util.simple_dominance` and
        :func:`moments.TwoLocus.Util.gene_based_dominance`.
    :param cache: If True, load the spectrum from the cache (see
        :func:`set_cache_path`) if it exists there, and otherwise compute it and save it
        to the cache. If False (default), compute the spectrum without using the cache.
    :returns: The equilibrium spectrum, as a :class:`moments.TwoLocus.TLSpectrum`.
    """
    if rho == None:
        print("Warning: no rho value set. Simulating with rho = 0.")
        rho = 0.0

    gamma = _make_floats(gamma)
    rho = _make_floats(rho)
    theta = _make_floats(theta)
    sel_params = _make_floats(sel_params)
    sel_params_general = _make_floats(sel_params_general)

    # check that only one selection model is given, and convert gamma to sel_params
    _, sel_params, sel_params_general = moments.TwoLocus.Integration._resolve_selection(
        gamma, sel_params, sel_params_general
    )

    if cache:
        eq_name = _equilibrium_cache_name(
            ns, rho, theta, sel_params, sel_params_general
        )
        try:
            F = moments.TwoLocus.TLSpectrum.from_file(eq_name)
            recompute = False
        except IOError:
            recompute = True

    if cache is False or recompute:
        F = moments.TwoLocus.Integration.steady_state(
            ns,
            rho=rho,
            theta=theta,
            sel_params=sel_params,
            sel_params_general=sel_params_general,
        )
        if cache:
            F.to_file(eq_name)
    return F


def two_epoch(params, ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    A two-epoch model, in which the population changes size instantaneously to nu
    (relative to the ancestral size) at time T in the past. T is given in units of 2Ne
    generations, where Ne is the ancestral effective size. A relative size of 1 implies
    no size change.

    :param params: Given as [nu, T].
    :param ns: The haploid sample size.
    :param rho: The population-size scaled recombination rate between the two loci,
        4*Ne*r. If not given, rho = 0 is used and a warning is printed.
    :param theta: The population-size scaled mutation rate at each locus, 4*Ne*u.
        Defaults to 1.
    :param gamma: The population-size scaled selection coefficient, 2*Ne*s, for
        additive selection with the same coefficient at both loci and no epistasis.
        This is shorthand for ``sel_params = [2 * gamma, gamma, gamma]``, so that the
        single-locus spectrum at each locus matches the single-site SFS with selection
        coefficient gamma and h = 1/2. For selection at only one locus, unequal
        selection, or epistasis, use ``sel_params`` instead. Cannot be combined with
        ``sel_params``.
    :param sel_params: Additive (within-locus) selection parameters for the AB, Ab, and
        aB haplotypes relative to ab, given as ``[gamma_AB, gamma_A, gamma_B]``, where
        each is 2*Ne*s for that haplotype. For example, selection only at the left locus
        is ``[gamma, gamma, 0]``. With ``gamma_AB = gamma_A + gamma_B`` there is no
        epistasis. See :func:`moments.TwoLocus.Util.additive_epistasis`.
    :returns: The two-locus frequency spectrum, as a
        :class:`moments.TwoLocus.TLSpectrum`.
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
    by a relative size change to nu2 that lasts for time T2. Times are in units of 2Ne
    generations, and sizes are relative to the ancestral Ne.

    :param params: Given as [nu1, nu2, T1, T2].
    :param ns: The haploid sample size.
    :param rho: The population-size scaled recombination rate between the two loci,
        4*Ne*r. If not given, rho = 0 is used and a warning is printed.
    :param theta: The population-size scaled mutation rate at each locus, 4*Ne*u.
        Defaults to 1.
    :param gamma: The population-size scaled selection coefficient, 2*Ne*s, for
        additive selection with the same coefficient at both loci and no epistasis.
        This is shorthand for ``sel_params = [2 * gamma, gamma, gamma]``, so that the
        single-locus spectrum at each locus matches the single-site SFS with selection
        coefficient gamma and h = 1/2. For selection at only one locus, unequal
        selection, or epistasis, use ``sel_params`` instead. Cannot be combined with
        ``sel_params``.
    :param sel_params: Additive (within-locus) selection parameters for the AB, Ab, and
        aB haplotypes relative to ab, given as ``[gamma_AB, gamma_A, gamma_B]``, where
        each is 2*Ne*s for that haplotype. For example, selection only at the left locus
        is ``[gamma, gamma, 0]``. With ``gamma_AB = gamma_A + gamma_B`` there is no
        epistasis. See :func:`moments.TwoLocus.Util.additive_epistasis`.
    :returns: The two-locus frequency spectrum, as a
        :class:`moments.TwoLocus.TLSpectrum`.
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
    An exponential growth model, which begins growth at time T ago, in units of 2Ne
    generations. The final size is given by nu, which is the size relative to the
    ancestral Ne.

    :param params: Given as [nu, T].
    :param ns: The haploid sample size.
    :param rho: The population-size scaled recombination rate between the two loci,
        4*Ne*r. If not given, rho = 0 is used and a warning is printed.
    :param theta: The population-size scaled mutation rate at each locus, 4*Ne*u.
        Defaults to 1.
    :param gamma: The population-size scaled selection coefficient, 2*Ne*s, for
        additive selection with the same coefficient at both loci and no epistasis.
        This is shorthand for ``sel_params = [2 * gamma, gamma, gamma]``, so that the
        single-locus spectrum at each locus matches the single-site SFS with selection
        coefficient gamma and h = 1/2. For selection at only one locus, unequal
        selection, or epistasis, use ``sel_params`` instead. Cannot be combined with
        ``sel_params``.
    :param sel_params: Additive (within-locus) selection parameters for the AB, Ab, and
        aB haplotypes relative to ab, given as ``[gamma_AB, gamma_A, gamma_B]``, where
        each is 2*Ne*s for that haplotype. For example, selection only at the left locus
        is ``[gamma, gamma, 0]``. With ``gamma_AB = gamma_A + gamma_B`` there is no
        epistasis. See :func:`moments.TwoLocus.Util.additive_epistasis`.
    :returns: The two-locus frequency spectrum, as a
        :class:`moments.TwoLocus.TLSpectrum`.
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
    F.integrate(nu_func, T, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    return F


def bottlegrowth(params, ns, rho=None, theta=1.0, gamma=None, sel_params=None):
    """
    A bottleneck followed by exponential growth. The population changes size to nuB
    at time T ago, and then has exponential size change to final size nuF. Time is
    in units of 2Ne generations, and sizes are relative to the ancestral Ne.

    :param params: Given as [nuB, nuF, T].
    :param ns: The haploid sample size.
    :param rho: The population-size scaled recombination rate between the two loci,
        4*Ne*r. If not given, rho = 0 is used and a warning is printed.
    :param theta: The population-size scaled mutation rate at each locus, 4*Ne*u.
        Defaults to 1.
    :param gamma: The population-size scaled selection coefficient, 2*Ne*s, for
        additive selection with the same coefficient at both loci and no epistasis.
        This is shorthand for ``sel_params = [2 * gamma, gamma, gamma]``, so that the
        single-locus spectrum at each locus matches the single-site SFS with selection
        coefficient gamma and h = 1/2. For selection at only one locus, unequal
        selection, or epistasis, use ``sel_params`` instead. Cannot be combined with
        ``sel_params``.
    :param sel_params: Additive (within-locus) selection parameters for the AB, Ab, and
        aB haplotypes relative to ab, given as ``[gamma_AB, gamma_A, gamma_B]``, where
        each is 2*Ne*s for that haplotype. For example, selection only at the left locus
        is ``[gamma, gamma, 0]``. With ``gamma_AB = gamma_A + gamma_B`` there is no
        epistasis. See :func:`moments.TwoLocus.Util.additive_epistasis`.
    :returns: The two-locus frequency spectrum, as a
        :class:`moments.TwoLocus.TLSpectrum`.
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
    F.integrate(nu_func, T, rho=rho, theta=theta, gamma=gamma, sel_params=sel_params)
    return F
