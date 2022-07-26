import numpy as np


"""
Heterozygosity and LD stat names
"""


def het_names(num_pops):
    """
    Returns the heterozygosity statistic representation names.

    :param int num_pops: Number of populations.
    """
    Hs = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            Hs.append("H_{0}_{1}".format(ii, jj))
    return Hs


def ld_names(num_pops):
    """
    Returns the LD statistic representation names.

    :param int num_pops: Number of populations.
    """
    Ys = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            Ys.append("DD_{0}_{1}".format(ii, jj))
    for ii in range(num_pops):
        for jj in range(num_pops):
            for kk in range(jj, num_pops):
                Ys.append("Dz_{0}_{1}_{2}".format(ii, jj, kk))
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            for kk in range(ii, num_pops):
                for ll in range(kk, num_pops):
                    if kk == ii == ll and jj != ii:
                        continue
                    if ii == kk and ll < jj:
                        continue
                    Ys.append("pi2_{0}_{1}_{2}_{3}".format(ii, jj, kk, ll))
    return Ys


def moment_names(num_pops):
    """
    Returns a tuple of length two with LD and heterozygosity moment names.

    :param int num_pops: Number of populations
    """
    hn = het_names(num_pops)
    yn = ld_names(num_pops)
    return (yn, hn)


##
## We need to map moments for split and rearrangement functions
##
mom_map = {}


def map_moment(mom):
    """
    There are repeated moments with equal expectations, so we collapse them into
    the same moment.

    :param str mom: The moment to map to its "canonical" name.
    """
    try:
        return mom_map[mom]
    except KeyError:
        if mom.split("_")[0] == "DD":
            pops = sorted([int(p) for p in mom.split("_")[1:]])
            mom_out = "DD_" + "_".join([str(p) for p in pops])
            mom_map[mom] = mom_out
        elif mom.split("_")[0] == "Dz":
            popD = mom.split("_")[1]
            popsz = sorted([int(p) for p in mom.split("_")[2:]])
            mom_out = "Dz_" + popD + "_" + "_".join([str(p) for p in popsz])
            mom_map[mom] = mom_out
        elif mom.split("_")[0] == "pi2":
            popsp = sorted([int(p) for p in mom.split("_")[1:3]])
            popsq = sorted([int(p) for p in mom.split("_")[3:]])
            ## pi2_2_2_1_1 -> pi2_1_1_2_2
            ## pi2_1_2_1_1 -> pi2_1_1_1_2,
            ## pi2_2_2_1_3 -> pi2_1_3_2_2
            if popsp[0] > popsq[0]:  # switch them
                mom_out = (
                    "pi2_"
                    + "_".join([str(p) for p in popsq])
                    + "_"
                    + "_".join([str(p) for p in popsp])
                )
            elif popsp[0] == popsq[0] and popsp[1] > popsq[1]:  # switch them
                mom_out = (
                    "pi2_"
                    + "_".join([str(p) for p in popsq])
                    + "_"
                    + "_".join([str(p) for p in popsp])
                )
            else:
                mom_out = (
                    "pi2_"
                    + "_".join([str(p) for p in popsp])
                    + "_"
                    + "_".join([str(p) for p in popsq])
                )
            mom_map[mom] = mom_out
        elif mom.split("_")[0] == "H":
            pops = sorted([int(p) for p in mom.split("_")[1:]])
            mom_out = "H_" + "_".join([str(p) for p in pops])
            mom_map[mom] = mom_out
        else:
            mom_out = mom
        mom_map[mom] = mom_out
        return mom_map[mom]


def perturb_params(params, fold=1, lower_bound=None, upper_bound=None):
    """
    Generate a perturbed set of parameters. Each element of params is randomly
    perturbed by the given factor of 2 up or down.

    :param list params: A list of input parameters.
    :param float fold: Number of factors of 2 to perturb by.
    :param list lower_bound: If not None, the resulting parameter set is adjusted
        to have all value greater than lower_bound. Must have equal length to
        ``params``.
    :param list upper_bound: If not None, the resulting parameter set is adjusted
        to have all value less than upper_bound. Must have equal length to
        ``params``.
    """
    pnew = params * 2 ** (fold * (2 * np.random.random(len(params)) - 1))
    if lower_bound is not None:
        for ii, bound in enumerate(lower_bound):
            if bound is None:
                lower_bound[ii] = -np.inf
        pnew = np.maximum(pnew, 1.01 * np.asarray(lower_bound))
    if upper_bound is not None:
        for ii, bound in enumerate(upper_bound):
            if bound is None:
                upper_bound[ii] = np.inf
        pnew = np.minimum(pnew, 0.99 * np.asarray(upper_bound))
    return pnew


def rescale_params(params, types, Ne=None, gens=1, uncerts=None, time_offset=0):
    """
    Rescale parameters to physical units, so that times are in generations or years,
    sizes in effective instead of relative sizes, and migration probabilities in
    per-generation units.

    For generation times of events to be correctly rescaled, times in the
    parameters list must be specified so that earlier epochs are earlier in the
    list, because we return rescaled cumulative times. All time parameters must
    refer to consecutive epochs. Epochs need not start at contemporary time, and
    we can specify the time offset using `time_offset`.

    If uncertainties are not given (`uncerts = None`), the return value is an
    array of rescaled parameters. If uncertainties are given, the return value
    has length two: the first entry is an array of rescaled parameters, and the
    second entry is an array of rescaled uncertainties.

    :param list params: List of parameters.
    :param list types: List of parameter types. Times are given by "T", sizes by "nu",
        effective size by "Ne", migration rates by "m", and fractions by "x" or "f".
    :param float Ne: The effective population size, typically as the last entry in
        ``params``.
    :param float gens: The generation time.
    :param list uncerts: List of uncertainties, same length as ``params``.
    :param time_offset: The amount of time added to each rescaled time point. This
        lets us have consecutive epochs that stop short of time 0 (final sampling
        time).
    :type time_offset: int or float
    """
    if Ne is not None and "Ne" in types:
        raise ValueError(
            "Ne must be passed as a keywork argument or "
            "specified in types, but not both."
        )
    elif Ne is None and "Ne" not in types:
        raise ValueError("Ne must be given or must exist in the types list")

    if types.count("Ne") > 1:
        raise ValueError("Ne can only be defined once in the parameters list")

    if Ne is None:
        Ne = params[types.index("Ne")]

    if len(params) != len(types):
        raise ValueError("types and params must have same length")

    if uncerts is not None:
        assert len(params) == len(uncerts)

    # rescale the params
    # go backwards to add times in reverse
    elapsed_t = 0
    rescaled_params = np.array([0.0 for _ in params])
    for ii, p in reversed(list(enumerate(params))):
        if types[ii] == "T":
            elapsed_t += p * 2 * Ne * gens
            rescaled_params[ii] = elapsed_t + time_offset
        elif types[ii] == "m":
            rescaled_params[ii] = p / 2 / Ne
        elif types[ii] == "nu":
            rescaled_params[ii] = p * Ne
        elif types[ii] in ["x", "f"]:
            rescaled_params[ii] = p
        elif types[ii] == "Ne":
            rescaled_params[ii] = Ne
        else:
            raise ValueError("Unrecognized parameter type", types[ii])

    if uncerts is None:
        return rescaled_params
    else:
        ## if uncerts are given
        # rescale the uncerts
        rescaled_uncerts = np.array([0.0 for _ in params])
        for ii, p in enumerate(uncerts):
            if types[ii] == "T":
                rescaled_uncerts[ii] = p * 2 * Ne * gens
            elif types[ii] == "m":
                rescaled_uncerts[ii] = p / 2 / Ne
            elif types[ii] == "nu":
                rescaled_uncerts[ii] = p * Ne
            else:
                rescaled_uncerts[ii] = p
        return rescaled_params, rescaled_uncerts
