import numpy as np


"""
Heterozygosity and LD stat names
"""

### handling data (splits, marginalization, admixture, ...)
def het_names(num_pops):
    Hs = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            Hs.append("H_{0}_{1}".format(ii + 1, jj + 1))
    return Hs


def ld_names(num_pops):
    Ys = []
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            Ys.append("DD_{0}_{1}".format(ii + 1, jj + 1))
    for ii in range(num_pops):
        for jj in range(num_pops):
            for kk in range(jj, num_pops):
                Ys.append("Dz_{0}_{1}_{2}".format(ii + 1, jj + 1, kk + 1))
    for ii in range(num_pops):
        for jj in range(ii, num_pops):
            for kk in range(ii, num_pops):
                for ll in range(kk, num_pops):
                    if kk == ii == ll and jj != ii:
                        continue
                    if ii == kk and ll < jj:
                        continue
                    Ys.append(
                        "pi2_{0}_{1}_{2}_{3}".format(ii + 1, jj + 1, kk + 1, ll + 1)
                    )
    return Ys


def moment_names(num_pops):
    """
    num_pops : number of populations, indexed [1,...,num_pops]
    """
    hn = het_names(num_pops)
    yn = ld_names(num_pops)
    return (yn, hn)


"""
We need to map moments for split and rearrangement functions
"""
mom_map = {}


def map_moment(mom):
    """
    
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
            ## pi2_2_2_1_1 -> pi2_1_1_2_2, pi2_1_2_1_1 -> pi2_1_1_1_2, pi2_2_2_1_3 -> pi2_1_3_2_2
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
    Generate a perturbed set of parameters.

    Each element of params is radomly perturbed <fold> factors of 2 up or down.
    fold: Number of factors of 2 to perturb by
    lower_bound: If not None, the resulting parameter set is adjusted to have 
                 all value greater than lower_bound.
    upper_bound: If not None, the resulting parameter set is adjusted to have 
                 all value less than upper_bound.
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


def rescale_params(params, types, Ne=None, gens=1, uncerts=None):
    """
    Times in params must be specified so that earlier epochs are earlier in the
    param list, because we return rescaled cumulative times.
    """
    if Ne is not None and "Ne" in types:
        raise ValueError(
            "Ne must be passed as a keywork argument or specified in types, but not both."
        )

    if Ne is None:
        Ne = params[types.index("Ne")]

    assert len(params) == len(types), "types and params must have same length"
    if uncerts is not None:
        assert len(params) == len(uncerts)

    # rescale the params
    # go backwards to add times in reverse
    elapsed_t = 0
    rescaled_params = [0 for _ in params]
    for ii, p in reversed(list(enumerate(params))):
        if types[ii] == "T":
            elapsed_t += p * 2 * Ne * gens
            rescaled_params[ii] = elapsed_t
        elif types[ii] == "m":
            rescaled_params[ii] = p / 2 / Ne
        elif types[ii] == "nu":
            rescaled_params[ii] = p * Ne
        else:
            rescaled_params[ii] = p

    if uncerts is None:
        return rescaled_params
    else:
        ## if uncerts are given
        # rescale the uncerts
        rescaled_uncerts = [0 for _ in params]
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
