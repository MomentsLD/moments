# Isolation-with-migration model with constant proportion of source demes
# that send migrants to destination demes

def IM_G(params, ns):
    """
    params = (nu10, nu20, nu1, nu2, T, m12, m21)
    ns = (n1, n2)

    m0_12 and m0_21 are constant proportions of the source populations that
    send migrants to the other population. We create migration functions that
    rescale these "source" migration rates to get the classical definition of
    migration rates, which is the proportion of the dest population that is
    made up of migrants from the source population.
    """
    nu10, nu20, nu1, nu2, T, m0_12, m0_21 = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

    # size functions, exponential in each population
    nu1_func = lambda t: nu10 * (nu1 / nu10) ** (t / T)
    nu2_func = lambda t: nu20 * (nu2 / nu20) ** (t / T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]

    # migration functions to keep proportion of source migrants constant
    m12_func = lambda t: m0_12 * nu2_func(t) / nu1_func(t)
    m21_func = lambda t: m0_21 * nu1_func(t) / nu2_func(t)
    m_func = lambda t: [[0, m12_func(t)], [m21_func(t), 0]]

    fs.integrate(nu_func, T, m=m_func)
    return fs
