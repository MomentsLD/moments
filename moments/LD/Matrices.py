import numpy as np
from . import Util

from scipy.sparse import csc_matrix

## matrices for new LDstats2 models (h and compressed ys)


### drift


def drift_h(num_pops, nus, frozen=None):
    if num_pops != len(nus):
        raise ValueError("number of pops must match length of nus.")

    # if any population is frozen, we set that population's size to something
    # extremely large, so that drift is effectively zero
    if frozen is not None:
        if num_pops != len(frozen):
            raise ValueError("length of 'frozen' must match number of pops.")
        for pid in range(num_pops):
            if frozen[pid] == True:
                nus[pid] = 1e30

    D = np.zeros(
        (int(num_pops * (num_pops + 1) / 2), int(num_pops * (num_pops + 1) / 2))
    )
    c = 0
    for ii in range(num_pops):
        D[c, c] = -1.0 / nus[ii]
        c += num_pops - ii
    return D


def drift_ld(num_pops, nus, frozen=None):
    if num_pops != len(nus):
        raise ValueError("number of pops must match length of nus.")

    # if any population is frozen, we set that population's size to something
    # extremely large, so that drift is effectively zero
    if frozen is not None:
        if num_pops != len(frozen):
            raise ValueError("length of 'frozen' must match number of pops.")
        for pid in range(num_pops):
            if frozen[pid] == True:
                nus[pid] = 1e30

    names = Util.ld_names(num_pops)
    row = []
    col = []
    data = []
    for ii, name in enumerate(names):
        mom = name.split("_")[0]
        pops = name.split("_")[1:]
        if mom == "DD":
            pop1, pop2 = [int(p) for p in pops]
            if pop1 == pop2:
                new_rows = [ii, ii, ii]
                new_cols = [
                    names.index("DD_{0}_{0}".format(pop1)),
                    names.index("Dz_{0}_{0}_{0}".format(pop1)),
                    names.index("pi2_{0}_{0}_{0}_{0}".format(pop1)),
                ]
                new_data = [
                    -3.0 / nus[pop1],
                    1.0 / nus[pop1],
                    1.0 / nus[pop1],
                ]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            else:
                row.append(ii)
                col.append(names.index("DD_{0}_{1}".format(pop1, pop2)))
                data.append(-1.0 / nus[pop1] - 1.0 / nus[pop2])
        elif mom == "Dz":
            pop1, pop2, pop3 = [int(p) for p in pops]
            if pop1 == pop2 == pop3:
                new_rows = [ii, ii]
                new_cols = [
                    names.index("DD_{0}_{0}".format(pop1)),
                    names.index("Dz_{0}_{0}_{0}".format(pop1)),
                ]
                new_data = [4.0 / nus[pop1], -5.0 / nus[pop1]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2:
                row.append(ii)
                col.append(names.index("Dz_{0}_{1}_{2}".format(pop1, pop2, pop3)))
                data.append(-3.0 / nus[pop1])
            elif pop1 == pop3:
                row.append(ii)
                col.append(names.index("Dz_{0}_{1}_{2}".format(pop1, pop2, pop3)))
                data.append(-3.0 / nus[pop1])
            elif pop2 == pop3:
                new_rows = [ii, ii]
                new_cols = [
                    names.index(Util.map_moment("DD_{0}_{1}".format(pop1, pop2))),
                    names.index("Dz_{0}_{1}_{1}".format(pop1, pop2)),
                ]
                new_data = [4.0 / nus[pop2], -1.0 / nus[pop1]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            else:  # all different
                row.append(ii)
                col.append(names.index("Dz_{0}_{1}_{2}".format(pop1, pop2, pop3)))
                data.append(-1.0 / nus[pop1])
        elif mom == "pi2":
            pop1, pop2, pop3, pop4 = [int(p) for p in pops]
            if pop1 == pop2 == pop3 == pop4:
                new_rows = [ii, ii]
                new_cols = [
                    names.index("Dz_{0}_{0}_{0}".format(pop1)),
                    names.index("pi2_{0}_{0}_{0}_{0}".format(pop1)),
                ]
                new_data = [1.0 / nus[pop1], -2.0 / nus[pop1]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2 == pop3:
                new_rows = [ii, ii]
                new_cols = [
                    names.index("Dz_{0}_{0}_{1}".format(pop1, pop4)),
                    names.index("pi2_{0}_{0}_{0}_{1}".format(pop1, pop4)),
                ]
                new_data = [1.0 / 2 / nus[pop1], -1.0 / nus[pop1]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2 == pop4:
                new_rows = [ii, ii]
                new_cols = [
                    names.index("Dz_{0}_{1}_{0}".format(pop1, pop3)),
                    names.index("pi2_{0}_{0}_{1}_{0}".format(pop1, pop3)),
                ]
                new_data = [1.0 / 2 / nus[pop1], -1.0 / nus[pop1]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop2 and pop3 == pop4:
                row.append(ii)
                col.append(names.index("pi2_{0}_{0}_{1}_{1}".format(pop1, pop3)))
                data.append(-1.0 / nus[pop1] - 1.0 / nus[pop3])
            elif pop1 == pop2:
                row.append(ii)
                col.append(names.index("pi2_{0}_{0}_{1}_{2}".format(pop1, pop3, pop4)))
                data.append(-1.0 / nus[pop1])
            elif pop1 == pop3 == pop4:
                new_rows = [ii, ii]
                new_cols = [
                    names.index(Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))),
                    names.index("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)),
                ]
                new_data = [1.0 / 2 / nus[pop1], -1.0 / nus[pop1]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop2 == pop3 == pop4:
                new_rows = [ii, ii]
                new_cols = [
                    names.index(Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))),
                    names.index("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)),
                ]
                new_data = [1.0 / 2 / nus[pop2], -1.0 / nus[pop2]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop3 and pop2 == pop4:
                new_rows = [ii, ii]
                new_cols = [
                    names.index("Dz_{0}_{1}_{1}".format(pop1, pop2)),
                    names.index("Dz_{1}_{0}_{0}".format(pop1, pop2)),
                ]
                new_data = [1.0 / 4 / nus[pop1], 1.0 / 4 / nus[pop2]]
                for r, c, d in zip(new_rows, new_cols, new_data):
                    row.append(r)
                    col.append(c)
                    data.append(d)
            elif pop1 == pop3:
                row.append(ii)
                col.append(
                    names.index(
                        Util.map_moment("Dz_{0}_{1}_{2}".format(pop1, pop2, pop4))
                    )
                )
                data.append(1.0 / 4 / nus[pop1])
            elif pop1 == pop4:
                row.append(ii)
                col.append(
                    names.index(
                        Util.map_moment("Dz_{0}_{1}_{2}".format(pop1, pop2, pop3))
                    )
                )
                data.append(1.0 / 4 / nus[pop1])
            elif pop2 == pop3:
                row.append(ii)
                col.append(
                    names.index(
                        Util.map_moment("Dz_{1}_{0}_{2}".format(pop1, pop2, pop4))
                    )
                )
                data.append(1.0 / 4 / nus[pop2])
            elif pop2 == pop4:
                row.append(ii)
                col.append(
                    names.index(
                        Util.map_moment("Dz_{1}_{0}_{2}".format(pop1, pop2, pop3))
                    )
                )
                data.append(1.0 / 4 / nus[pop2])
            elif pop3 == pop4:
                row.append(ii)
                col.append(names.index("pi2_{0}_{1}_{2}_{2}".format(pop1, pop2, pop3)))
                data.append(-1.0 / nus[pop3])
            else:
                if len(set([pop1, pop2, pop3, pop4])) < 4:
                    print("oh no")
                    print(pop1, pop2, pop3, pop4)

    return csc_matrix((data, (row, col)), shape=(len(names), len(names)))


### mutation
def mutation_h(num_pops, theta, frozen=None, selfing=None):
    if frozen is None and selfing is None:
        return theta * np.ones(int(num_pops * (num_pops + 1) / 2))
    else:
        U = np.zeros(int(num_pops * (num_pops + 1) / 2))
        if selfing is None:
            selfing = [0] * num_pops
        else:
            for ii, f in enumerate(selfing):
                if f == None:
                    selfing[ii] = 0
        if frozen is None:
            frozen = [False] * num_pops
        c = 0
        for ii in range(num_pops):
            for jj in range(ii, num_pops):
                if not frozen[ii]:
                    U[c] += theta / 2.0 * (1 - selfing[ii] / 2.0)
                if not frozen[jj]:
                    U[c] += theta / 2.0 * (1 - selfing[jj] / 2.0)
                c += 1
        return U


def mutation_ld(num_pops, theta, frozen=None, selfing=None):
    names_ld, names_h = Util.moment_names(num_pops)
    row = []
    col = []
    data = []
    if hasattr(selfing, "__len__"):
        if np.all([s is None for s in selfing]):
            selfing = None

    if frozen is None and selfing is None:
        for ii, mom in enumerate(names_ld):
            name = mom.split("_")[0]
            if name == "pi2":
                hmomp = "H_" + mom.split("_")[1] + "_" + mom.split("_")[2]
                hmomq = "H_" + mom.split("_")[3] + "_" + mom.split("_")[4]
                if hmomp == hmomq:
                    row.append(ii)
                    col.append(names_h.index(hmomp))
                    data.append(theta / 2.0)
                else:
                    row.append(ii)
                    row.append(ii)
                    col.append(names_h.index(hmomp))
                    col.append(names_h.index(hmomq))
                    data.append(theta / 4.0)
                    data.append(theta / 4.0)

    else:
        thetas = [theta for _ in range(num_pops)]

        if selfing is not None:
            for i, s in enumerate(selfing):
                thetas[i] = thetas[i] * (1 - s / 2.0)

        if frozen is not None:
            for pid in range(num_pops):
                if frozen[pid] == True:
                    thetas[pid] = 0.0

        for ii, mom in enumerate(names_ld):
            name = mom.split("_")[0]
            if name == "pi2":
                i, j, k, l = [int(x) for x in mom.split("_")[1:]]
                hmomp = "H_" + str(i) + "_" + str(j)
                hmomq = "H_" + str(k) + "_" + str(l)
                if hmomp == hmomq:
                    # i=k and j=l
                    row.append(ii)
                    col.append(names_h.index(hmomp))
                    data.append((thetas[i] / 2.0 + thetas[j] / 2.0) / 2.0)
                else:
                    row.append(ii)
                    col.append(names_h.index(hmomp))  # check if k and l are frozen
                    data.append((thetas[k] / 2.0 + thetas[l] / 2.0) / 4.0)
                    row.append(ii)
                    col.append(names_h.index(hmomq))  # check if i and j are frozen
                    data.append((thetas[i] / 2.0 + thetas[j] / 2.0) / 4.0)

    return csc_matrix((data, (row, col)), shape=(len(names_ld), len(names_h)))


### recombination


def recombination(num_pops, r, frozen=None, selfing=None):
    if hasattr(selfing, "__len__"):
        if np.all([s is None for s in selfing]):
            selfing = None

    names = Util.ld_names(num_pops)
    row = list(
        range(int(num_pops * (num_pops + 1) / 2 + num_pops ** 2 * (num_pops + 1) / 2))
    )
    col = list(
        range(int(num_pops * (num_pops + 1) / 2 + num_pops ** 2 * (num_pops + 1) / 2))
    )

    if frozen is None and selfing is None:
        data = [-1.0 * r] * int(num_pops * (num_pops + 1) / 2) + [-r / 2.0] * int(
            num_pops ** 2 * (num_pops + 1) / 2
        )
    else:
        rs = [r for _ in range(num_pops)]
        if selfing is not None:
            for i, s in enumerate(selfing):
                rs[i] = rs[i] * (1 - s)

        if frozen is not None:
            for pid in range(num_pops):
                if frozen[pid]:
                    rs[pid] = 0

        data = []
        for name in names:
            if name.split("_")[0] == "DD":
                pop1 = int(name.split("_")[1])
                pop2 = int(name.split("_")[2])
                data.append(-1.0 / 2 * rs[pop1] - 1.0 / 2 * rs[pop2])
            elif name.split("_")[0] == "Dz":
                pop1 = int(name.split("_")[1])
                data.append(-1.0 / 2 * rs[pop1])
            else:
                continue
    return csc_matrix((data, (row, col)), shape=(len(names), len(names)))


### migration


def migration_h(num_pops, mig_mat, frozen=None):
    """
    mig_mat has the form [[0, m12, m13, ..., m1n], ..., [mn1, mn2, ..., 0]]
    Note that m12 is the probability that a lineage in deme 1 had its parent
    in deme 2, to be consisten with moments (fs).
    """
    if frozen is not None:
        for pid in range(num_pops):
            if frozen[pid] == True:
                for pid2 in range(num_pops):
                    mig_mat[pid][pid2] = 0
                    mig_mat[pid2][pid] = 0

    Hs = Util.het_names(num_pops)
    # avoid calling Hs.index() and Util.map_moment
    H_indexes = {}
    for i, H in enumerate(Hs):
        ii, jj = [int(_) for _ in H.split("_")[1:]]
        H_indexes[(ii, jj)] = i
        H_indexes[(jj, ii)] = i

    M = np.zeros((len(Hs), len(Hs)))
    for ii, H in enumerate(Hs):
        pop1, pop2 = [int(f) for f in H.split("_")[1:]]
        if pop1 == pop2:
            for jj in range(num_pops):
                if jj == pop1:
                    continue
                else:
                    M[ii, ii] -= 2 * mig_mat[pop1][jj]
                    M[ii, H_indexes[(pop1, jj)]] += 2 * mig_mat[pop1][jj]
        else:
            for jj in range(num_pops):
                if jj == pop1:
                    continue
                else:
                    M[ii, ii] -= mig_mat[pop1][jj]
                    M[ii, H_indexes[(pop2, jj)]] += mig_mat[pop1][jj]
            for jj in range(num_pops):
                if jj == pop2:
                    continue
                else:
                    M[ii, ii] -= mig_mat[pop2][jj]
                    M[ii, H_indexes[(pop1, jj)]] += mig_mat[pop2][jj]

    return M


def migration_ld(num_pops, mig_mat, frozen=None):
    if frozen is not None:
        for pid in range(num_pops):
            if frozen[pid] == True:
                for pid2 in range(num_pops):
                    mig_mat[pid][pid2] = 0
                    mig_mat[pid2][pid] = 0

    Ys = Util.ld_names(num_pops)
    Y_indexes = {}
    for i, Y in enumerate(Ys):
        stat = Y.split("_")[0]
        inds = [int(_) for _ in Y.split("_")[1:]]
        Y_indexes.setdefault(stat, {})
        if stat == "DD":
            ii, jj = inds
            Y_indexes[stat][(ii, jj)] = i
            Y_indexes[stat][(jj, ii)] = i
        elif stat == "Dz":
            ii, jj, kk = inds
            Y_indexes[stat][(ii, jj, kk)] = i
            Y_indexes[stat][(ii, kk, jj)] = i
        elif stat == "pi2":
            ii, jj, kk, ll = inds
            Y_indexes[stat][(ii, jj, kk, ll)] = i
            Y_indexes[stat][(ii, jj, ll, kk)] = i
            Y_indexes[stat][(jj, ii, kk, ll)] = i
            Y_indexes[stat][(jj, ii, ll, kk)] = i
            Y_indexes[stat][(kk, ll, ii, jj)] = i
            Y_indexes[stat][(ll, kk, ii, jj)] = i
            Y_indexes[stat][(kk, ll, jj, ii)] = i
            Y_indexes[stat][(ll, kk, jj, ii)] = i

    M = np.zeros((len(Ys), len(Ys)))
    for ii, mom in enumerate(Ys):
        name = mom.split("_")[0]
        pops = [int(p) for p in mom.split("_")[1:]]
        if name == "DD":
            pop1, pop2 = pops
            if pop1 == pop2:
                for jj in range(num_pops):
                    if jj != pop1:
                        M[ii, Y_indexes["DD"][(pop1, pop1)]] -= 2 * mig_mat[pop1][jj]
                        M[ii, Y_indexes["DD"][(pop1, jj)]] += 2 * mig_mat[pop1][jj]
                        M[ii, Y_indexes["Dz"][(pop1, pop1, pop1)]] += (
                            mig_mat[pop1][jj] / 2
                        )
                        M[ii, Y_indexes["Dz"][(pop1, pop1, jj)]] -= (
                            mig_mat[pop1][jj] / 2
                        )
                        M[ii, Y_indexes["Dz"][(pop1, jj, pop1)]] -= (
                            mig_mat[pop1][jj] / 2
                        )
                        M[ii, Y_indexes["Dz"][(pop1, jj, jj)]] += mig_mat[pop1][jj] / 2

            else:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["DD"][(pop1, pop2)]] -= mig_mat[pop1][kk]
                        M[ii, Y_indexes["DD"][(kk, pop2)]] += mig_mat[pop1][kk]
                        M[ii, Y_indexes["Dz"][(pop2, pop1, pop1)]] += (
                            mig_mat[pop1][kk] / 4
                        )
                        M[ii, Y_indexes["Dz"][(pop2, pop1, kk)]] -= (
                            mig_mat[pop1][kk] / 4
                        )
                        M[ii, Y_indexes["Dz"][(pop2, kk, pop1)]] -= (
                            mig_mat[pop1][kk] / 4
                        )
                        M[ii, Y_indexes["Dz"][(pop2, kk, kk)]] += mig_mat[pop1][kk] / 4

                    if kk != pop2:
                        M[ii, Y_indexes["DD"][(pop1, pop2)]] -= mig_mat[pop2][kk]
                        M[ii, Y_indexes["DD"][(pop1, kk)]] += mig_mat[pop2][kk]
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop2)]] += (
                            mig_mat[pop2][kk] / 4
                        )
                        M[ii, Y_indexes["Dz"][(pop1, pop2, kk)]] -= (
                            mig_mat[pop2][kk] / 4
                        )
                        M[ii, Y_indexes["Dz"][(pop1, kk, pop2)]] -= (
                            mig_mat[pop2][kk] / 4
                        )
                        M[ii, Y_indexes["Dz"][(pop1, kk, kk)]] += mig_mat[pop2][kk] / 4

        elif name == "Dz":
            pop1, pop2, pop3 = pops
            if pop1 == pop2 == pop3:
                for jj in range(num_pops):
                    if jj != pop1:
                        M[ii, Y_indexes["Dz"][(pop1, pop1, pop1)]] -= (
                            3 * mig_mat[pop1][jj]
                        )
                        M[ii, Y_indexes["Dz"][(pop1, pop1, jj)]] += mig_mat[pop1][jj]
                        M[ii, Y_indexes["Dz"][(pop1, jj, pop1)]] += mig_mat[pop1][jj]
                        M[ii, Y_indexes["Dz"][(jj, pop1, pop1)]] += mig_mat[pop1][jj]
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, pop1)]] += (
                            4 * mig_mat[pop1][jj]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, jj)]] -= (
                            4 * mig_mat[pop1][jj]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, jj, pop1, pop1)]] -= (
                            4 * mig_mat[pop1][jj]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, jj, pop1, jj)]] += (
                            4 * mig_mat[pop1][jj]
                        )

            elif pop1 == pop2:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["Dz"][(pop1, pop1, pop3)]] -= (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["Dz"][(kk, pop1, pop3)]] += mig_mat[pop1][kk]
                        M[ii, Y_indexes["Dz"][(pop1, kk, pop3)]] += mig_mat[pop1][kk]
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, pop3)]] += (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, kk)]] -= (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop1, pop3)]] -= (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop3, kk)]] += (
                            4 * mig_mat[pop1][kk]
                        )
                    if kk != pop3:
                        M[ii, Y_indexes["Dz"][(pop1, pop1, pop3)]] -= mig_mat[pop3][kk]
                        M[ii, Y_indexes["Dz"][(pop1, pop1, kk)]] += mig_mat[pop3][kk]

            elif pop1 == pop3:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop1)]] -= (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["Dz"][(kk, pop2, pop1)]] += mig_mat[pop1][kk]
                        M[ii, Y_indexes["Dz"][(pop1, pop2, kk)]] += mig_mat[pop1][kk]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop1)]] += (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, kk)]] -= (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop1, pop1)]] -= (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop1, kk)]] += (
                            4 * mig_mat[pop1][kk]
                        )
                    if kk != pop2:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop1)]] -= mig_mat[pop2][kk]
                        M[ii, Y_indexes["Dz"][(pop1, kk, pop1)]] += mig_mat[pop2][kk]

            elif pop2 == pop3:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop2)]] -= mig_mat[pop1][kk]
                        M[ii, Y_indexes["Dz"][(kk, pop2, pop2)]] += mig_mat[pop1][kk]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop2)]] += (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, kk)]] -= (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop1, pop2)]] -= (
                            4 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop2, kk)]] += (
                            4 * mig_mat[pop1][kk]
                        )
                    if kk != pop2:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop2)]] -= (
                            2 * mig_mat[pop2][kk]
                        )
                        M[ii, Y_indexes["Dz"][(pop1, pop2, kk)]] += mig_mat[pop2][kk]
                        M[ii, Y_indexes["Dz"][(pop1, kk, pop2)]] += mig_mat[pop2][kk]

            else:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop3)]] -= mig_mat[pop1][ll]
                        M[ii, Y_indexes["Dz"][(ll, pop2, pop3)]] += mig_mat[pop1][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop3)]] += (
                            4 * mig_mat[pop1][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, ll)]] -= (
                            4 * mig_mat[pop1][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop1, pop3)]] -= (
                            4 * mig_mat[pop1][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop3, ll)]] += (
                            4 * mig_mat[pop1][ll]
                        )
                    if ll != pop2:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop3)]] -= mig_mat[pop2][ll]
                        M[ii, Y_indexes["Dz"][(pop1, ll, pop3)]] += mig_mat[pop2][ll]
                    if ll != pop3:
                        M[ii, Y_indexes["Dz"][(pop1, pop2, pop3)],] -= mig_mat[
                            pop3
                        ][ll]
                        M[ii, Y_indexes["Dz"][(pop1, pop2, ll)],] += mig_mat[
                            pop3
                        ][ll]

        elif name == "pi2":
            pop1, pop2, pop3, pop4 = pops
            if pop1 == pop2 == pop3 == pop4:
                for jj in range(num_pops):
                    if jj != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, pop1)]] -= (
                            4 * mig_mat[pop1][jj]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, jj)]] += (
                            2 * mig_mat[pop1][jj]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, jj, pop1, pop1)]] += (
                            2 * mig_mat[pop1][jj]
                        )

            elif pop1 == pop2 == pop3:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, pop4)]] -= (
                            3 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop1, pop4)]] += (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop4, kk)]] += mig_mat[
                            pop1
                        ][kk]
                    if kk != pop4:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, pop4)]] -= mig_mat[
                            pop4
                        ][kk]
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop1, kk)]] += mig_mat[
                            pop4
                        ][kk]

            elif pop1 == pop2 == pop4:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop1)]] -= (
                            3 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop3, pop1)]] += (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, kk)]] += mig_mat[
                            pop1
                        ][kk]
                    if kk != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop1)]] -= mig_mat[
                            pop3
                        ][kk]
                        M[ii, Y_indexes["pi2"][(pop1, pop1, kk, pop1)]] += mig_mat[
                            pop3
                        ][kk]

            elif pop1 == pop2 and pop3 == pop4:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop3)]] -= (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop3, pop3)]] += (
                            2 * mig_mat[pop1][kk]
                        )
                    if kk != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop3)]] -= (
                            2 * mig_mat[pop3][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, kk)]] += (
                            2 * mig_mat[pop3][kk]
                        )

            elif pop1 == pop2:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop4)]] -= (
                            2 * mig_mat[pop1][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, ll, pop3, pop4)]] += (
                            2 * mig_mat[pop1][ll]
                        )
                    if ll != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop4)]] -= mig_mat[
                            pop3
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop4, ll)]] += mig_mat[
                            pop3
                        ][ll]
                    if ll != pop4:
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, pop4)]] -= mig_mat[
                            pop4
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop1, pop3, ll)]] += mig_mat[
                            pop4
                        ][ll]

            elif pop1 == pop3 == pop4:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop1)]] -= (
                            3 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, kk)]] += (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop1, pop1)]] += mig_mat[
                            pop1
                        ][kk]
                    if kk != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop1)]] -= mig_mat[
                            pop2
                        ][kk]
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop1, pop1)]] += mig_mat[
                            pop2
                        ][kk]

            elif pop2 == pop3 == pop4:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, pop2)]] -= mig_mat[
                            pop1
                        ][kk]
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop2, pop2)]] += mig_mat[
                            pop1
                        ][kk]
                    if kk != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, pop2)]] -= (
                            3 * mig_mat[pop2][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, kk)]] += (
                            2 * mig_mat[pop2][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop2, pop2)]] += mig_mat[
                            pop2
                        ][kk]

            elif pop1 == pop3 and pop2 == pop4:
                for kk in range(num_pops):
                    if kk != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop2)]] -= (
                            2 * mig_mat[pop1][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, kk)]] += mig_mat[
                            pop1
                        ][kk]
                        M[ii, Y_indexes["pi2"][(pop2, kk, pop1, pop2)]] += mig_mat[
                            pop1
                        ][kk]
                    if kk != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop2)]] -= (
                            2 * mig_mat[pop2][kk]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, kk)]] += mig_mat[
                            pop2
                        ][kk]
                        M[ii, Y_indexes["pi2"][(pop1, kk, pop1, pop2)]] += mig_mat[
                            pop2
                        ][kk]

            elif pop1 == pop3:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop4)]] -= (
                            2 * mig_mat[pop1][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop4, ll)]] += mig_mat[
                            pop1
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop1, pop4)]] += mig_mat[
                            pop1
                        ][ll]
                    if ll != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop4)]] -= mig_mat[
                            pop2
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, ll, pop1, pop4)]] += mig_mat[
                            pop2
                        ][ll]
                    if ll != pop4:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, pop4)]] -= mig_mat[
                            pop4
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop1, ll)]] += mig_mat[
                            pop4
                        ][ll]

            elif pop1 == pop4:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop1)]] -= (
                            2 * mig_mat[pop1][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop3, pop1)]] += mig_mat[
                            pop1
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, ll)]] += mig_mat[
                            pop1
                        ][ll]
                    if ll != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop1)]] -= mig_mat[
                            pop2
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, ll, pop3, pop1)]] += mig_mat[
                            pop2
                        ][ll]
                    if ll != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop1)]] -= mig_mat[
                            pop3
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, ll, pop1)]] += mig_mat[
                            pop3
                        ][ll]

            elif pop2 == pop3:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, pop4)]] -= mig_mat[
                            pop1
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop2, pop4)]] += mig_mat[
                            pop1
                        ][ll]
                    if ll != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, pop4)]] -= (
                            2 * mig_mat[pop2][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, ll, pop2, pop4)]] += mig_mat[
                            pop2
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop4, ll)]] += mig_mat[
                            pop2
                        ][ll]
                    if ll != pop4:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, pop4)]] -= mig_mat[
                            pop4
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop2, ll)]] += mig_mat[
                            pop4
                        ][ll]

            elif pop2 == pop4:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop2)]] -= mig_mat[
                            pop1
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop3, pop2)]] += mig_mat[
                            pop1
                        ][ll]
                    if ll != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop2)]] -= (
                            2 * mig_mat[pop2][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, ll, pop3, pop2)]] += mig_mat[
                            pop2
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, ll)]] += mig_mat[
                            pop2
                        ][ll]
                    if ll != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop2)]] -= mig_mat[
                            pop3
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, ll, pop2)]] += mig_mat[
                            pop3
                        ][ll]

            elif pop3 == pop4:
                for ll in range(num_pops):
                    if ll != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop3)]] -= mig_mat[
                            pop1
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop2, ll, pop3, pop3)]] += mig_mat[
                            pop1
                        ][ll]
                    if ll != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop3)]] -= mig_mat[
                            pop2
                        ][ll]
                        M[ii, Y_indexes["pi2"][(pop1, ll, pop3, pop3)]] += mig_mat[
                            pop2
                        ][ll]
                    if ll != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop3)]] -= (
                            2 * mig_mat[pop3][ll]
                        )
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, ll)]] += (
                            2 * mig_mat[pop3][ll]
                        )

            else:
                if len(set([pop1, pop2, pop3, pop4])) != 4:
                    print("fucked up again")
                for ss in range(num_pops):
                    if ss != pop1:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop4)]] -= mig_mat[
                            pop1
                        ][ss]
                        M[ii, Y_indexes["pi2"][(ss, pop2, pop3, pop4)]] += mig_mat[
                            pop1
                        ][ss]
                    if ss != pop2:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop4)]] -= mig_mat[
                            pop2
                        ][ss]
                        M[ii, Y_indexes["pi2"][(pop1, ss, pop3, pop4)]] += mig_mat[
                            pop2
                        ][ss]
                    if ss != pop3:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop4)]] -= mig_mat[
                            pop3
                        ][ss]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, ss, pop4)]] += mig_mat[
                            pop3
                        ][ss]
                    if ss != pop4:
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, pop4)]] -= mig_mat[
                            pop4
                        ][ss]
                        M[ii, Y_indexes["pi2"][(pop1, pop2, pop3, ss)]] += mig_mat[
                            pop4
                        ][ss]

    return csc_matrix(M)


def admix_h(num_pops, pop1, pop2, f):
    moms_from = Util.moment_names(num_pops)[1]
    moms_to = Util.moment_names(num_pops + 1)[1]
    A = np.zeros((len(moms_to), len(moms_from)))
    for ii, mom_to in enumerate(moms_to):
        if mom_to in moms_from:  # doesn't involve new pop (unchanged)
            A[ii, moms_from.index(mom_to)] = 1
        else:  # all moments are of the form H_k_new, k in [0,...,new] (new = num_pops)
            i1 = int(mom_to.split("_")[1])
            i2 = int(mom_to.split("_")[2])
            if i2 != num_pops:
                raise ValueError("This is unexpected... i2 should have been num_pops.")
            if i1 == i2 == num_pops:  # H_new_new
                A[ii, moms_from.index(Util.map_moment("H_{0}_{0}".format(pop1)))] = (
                    f ** 2
                )
                A[
                    ii, moms_from.index(Util.map_moment("H_{0}_{1}".format(pop1, pop2)))
                ] = (2 * f * (1 - f))
                A[ii, moms_from.index(Util.map_moment("H_{0}_{0}".format(pop2)))] = (
                    1 - f
                ) ** 2
            elif i1 == pop1:  # H_pop1_new
                A[ii, moms_from.index(Util.map_moment("H_{0}_{0}".format(pop1)))] = f
                A[
                    ii, moms_from.index(Util.map_moment("H_{0}_{1}".format(pop1, pop2)))
                ] = (1 - f)
            elif i1 == pop2:  # H_pop2_new
                A[
                    ii, moms_from.index(Util.map_moment("H_{0}_{1}".format(pop1, pop2)))
                ] = f
                A[ii, moms_from.index(Util.map_moment("H_{0}_{0}".format(pop2)))] = (
                    1 - f
                )
            else:  # H_non-source_new
                A[
                    ii, moms_from.index(Util.map_moment("H_{0}_{1}".format(pop1, i1)))
                ] = f
                A[
                    ii, moms_from.index(Util.map_moment("H_{0}_{1}".format(pop2, i1)))
                ] = (1 - f)
    return A


def admix_ld(num_pops, pop1, pop2, f):
    moms_from = Util.moment_names(num_pops)[0]
    moms_to = Util.moment_names(num_pops + 1)[0]
    A = np.zeros((len(moms_to), len(moms_from)))
    for ii, mom_to in enumerate(moms_to):
        if mom_to in moms_from:  # doesn't involve new pop (unchanged)
            A[ii, moms_from.index(mom_to)] = 1
        else:  # moments are either DD, Dz, or pi2. we handle each in turn
            mom_name = mom_to.split("_")[0]
            if mom_name == "DD":
                i1 = int(mom_to.split("_")[1])
                i2 = int(mom_to.split("_")[2])

                if i1 == i2 == num_pops:  # DD_new_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        2 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 2

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 2 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 2 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 2 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 2 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 2 * f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 2 * f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 2 * f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 2 * f * (1 - f) ** 2
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -2 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -2 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -2 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -2 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2 * (1 - f) ** 2
                    )

                elif i1 == pop1:  # DD_pop1_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 4 * f * (1 - f)
                    )

                elif i1 == pop2:  # DD_pop2_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1.0 / 4 * f * (1 - f)
                    )

                else:  # DD_non-source_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{0}_{2}".format(pop1, pop2, i1))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("DD_{1}_{2}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{0}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{0}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        -1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{1}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        -1.0 / 4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{1}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        1.0 / 4 * f * (1 - f)
                    )

            elif mom_name == "Dz":
                i1 = int(mom_to.split("_")[1])
                i2 = int(mom_to.split("_")[2])
                i3 = int(mom_to.split("_")[3])

                if i1 == i2 == i3 == num_pops:  # Dz_new_new_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 3
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += f ** 2 * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f ** 2 * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += f ** 2 * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 3

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 3 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f) * (1 - 2 * f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f) ** 2 * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f) ** 2 * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f) ** 3
                    )

                elif i1 == pop1 and i2 == i3 == num_pops:  # Dz_pop1_new_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 2

                elif i1 == pop2 and i2 == i3 == num_pops:  # Dz_pop2_new_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 2

                elif i2 == i3 == num_pops:  # Dz_non-source_new_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{0}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{0}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{1}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{1}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += (1 - f) ** 2

                elif i1 == i3 == num_pops and i2 == pop1:  # Dz_new_pop1_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 2

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f) ** 2
                    )

                elif i1 == i3 == num_pops and i2 == pop2:  # Dz_ne2_pop2_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 2

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f) ** 2
                    )

                elif i1 == i3 == num_pops:  # Dz_new_non-source_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{0}".format(pop1, pop2, i2))
                        ),
                    ] += (
                        f ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{1}".format(pop1, pop2, i2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{0}".format(pop1, pop2, i2))
                        ),
                    ] += f * (1 - f)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{1}".format(pop1, pop2, i2))
                        ),
                    ] += (1 - f) ** 2

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f ** 2 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f) * (1 - 2 * f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f) ** 2
                    )

                elif i1 == num_pops and i2 == pop1 and i3 == pop1:  # Dz_new_pop1_pop1
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i2 == pop1 and i3 == pop2:  # Dz_new_pop1_pop2
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i2 == pop2 and i3 == pop1:  # Dz_new_pop2_pop1
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i2 == pop2 and i3 == pop2:  # Dz_new_pop2_pop2
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i2 == pop1:  # Dz_new_pop1_non
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{2}".format(pop1, pop2, i3))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{2}".format(pop1, pop2, i3))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{0}_{0}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{0}_{1}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i2 == pop2:  # Dz_new_pop2_non
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{2}".format(pop1, pop2, i3))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{2}".format(pop1, pop2, i3))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{1}_{0}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{1}_{1}_{2}".format(pop1, pop2, i3)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i3 == pop1:  # Dz_new_non_pop1
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{0}".format(pop1, pop2, i2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{0}".format(pop1, pop2, i2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops and i3 == pop2:  # Dz_new_non_pop2
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{1}".format(pop1, pop2, i2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{1}".format(pop1, pop2, i2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif (
                    i1 == num_pops and i2 == i3
                ):  # Dz_new_non_non (same non-source pop)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{2}".format(pop1, pop2, i2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{2}".format(pop1, pop2, i2))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{2}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{1}_{2}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{2}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{1}_{2}".format(pop1, pop2, i2)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == num_pops:  # Dz_new_non1_non2 (different non-source pops)
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{3}".format(pop1, pop2, i2, i3))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{3}".format(pop1, pop2, i2, i3))
                        ),
                    ] += (
                        1 - f
                    )

                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{0}_{3}".format(pop1, pop2, i2, i3)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{0}_{2}_{1}_{3}".format(pop1, pop2, i2, i3)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{0}_{3}".format(pop1, pop2, i2, i3)
                            )
                        ),
                    ] += (
                        -4 * f * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment(
                                "pi2_{1}_{2}_{1}_{3}".format(pop1, pop2, i2, i3)
                            )
                        ),
                    ] += (
                        4 * f * (1 - f)
                    )

                elif i1 == pop1 and i2 == pop1 and i3 == num_pops:  # Dz_pop1_pop1_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                elif i1 == pop1 and i2 == pop2 and i3 == num_pops:  # Dz_pop1_pop2_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                elif i1 == pop1 and i3 == num_pops:  # Dz_pop1_non_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{0}".format(pop1, pop2, i2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{0}_{2}_{1}".format(pop1, pop2, i2))
                        ),
                    ] += (
                        1 - f
                    )

                elif i1 == pop2 and i2 == pop1 and i3 == num_pops:  # Dz_pop2_pop1_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                elif i1 == pop2 and i2 == pop2 and i3 == num_pops:  # Dz_pop2_pop2_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{0}".format(pop1, pop2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        1 - f
                    )

                elif i1 == pop2 and i3 == num_pops:  # Dz_pop2_non_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{0}".format(pop1, pop2, i2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{1}_{2}_{1}".format(pop1, pop2, i2))
                        ),
                    ] += (
                        1 - f
                    )

                elif i2 == pop1 and i3 == num_pops:  # Dz_non_pop1_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{0}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{0}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        1 - f
                    )

                elif i2 == pop2 and i3 == num_pops:  # Dz_non_pop2_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{1}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{1}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        1 - f
                    )

                elif i1 == i2 and i3 == num_pops:  # Dz_non_non_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{2}_{0}".format(pop1, pop2, i1))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{2}_{1}".format(pop1, pop2, i1))
                        ),
                    ] += (
                        1 - f
                    )

                elif i3 == num_pops:  # Dz_non1_non2_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{3}_{0}".format(pop1, pop2, i1, i2))
                        ),
                    ] += f
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("Dz_{2}_{3}_{1}".format(pop1, pop2, i1, i2))
                        ),
                    ] += (
                        1 - f
                    )

                else:
                    print("missed a Dz: ", mom_to)

            elif mom_name == "pi2":
                i1 = int(mom_to.split("_")[1])
                i2 = int(mom_to.split("_")[2])
                i3 = int(mom_to.split("_")[3])
                i4 = int(mom_to.split("_")[4])

                if i1 == i2 == i3 == i4 == num_pops:  # pi2_new_new_new_new
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 4
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        2 * f ** 3 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{0}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        2 * f ** 3 * (1 - f)
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        4 * f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{0}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        2 * f * (1 - f) ** 3
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{0}".format(pop1, pop2))
                        ),
                    ] += (
                        f ** 2 * (1 - f) ** 2
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{0}_{1}".format(pop1, pop2))
                        ),
                    ] += (
                        2 * f * (1 - f) ** 3
                    )
                    A[
                        ii,
                        moms_from.index(
                            Util.map_moment("pi2_{1}_{1}_{1}_{1}".format(pop1, pop2))
                        ),
                    ] += (1 - f) ** 4

                elif i2 == i3 == i4 == num_pops:
                    if i1 == pop1:  # pi2_pop1_new_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 3
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f ** 2 * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f * (1 - f) ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f ** 2 * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f) ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 3

                    elif i1 == pop2:  # pi2_pop2_new_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 3
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f ** 2 * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f * (1 - f) ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f ** 2 * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f) ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 3

                    else:  # pi2_non-source_new_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 3
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            2 * f ** 2 * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f * (1 - f) ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f ** 2 * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f) ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 3

                elif i3 == i4 == num_pops:
                    if i1 == i2 == pop1:  # pi2_pop1_pop1_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop1 and i2 == pop2:  # pi2_pop1_pop2_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop1:  # pi2_pop1_non-source_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop2 and i2 == pop1:  # pi2_pop2_pop1_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop2 and i2 == pop2:  # pi2_pop2_pop2_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop2:  # pi2_pop2_non-source_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i2 == pop1:  # pi2_non-source_pop1_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i2 == pop2:  # pi2_non-source_pop2_new_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == i2:  # pi2_non_non_new_new (non-source pops are the same)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 2

                    else:  # pi2_non1_non2_new_new (differenc=t non-source pops)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{0}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{1}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (
                            2 * f * (1 - f)
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{1}_{1}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (1 - f) ** 2

                elif i2 == num_pops and i4 == num_pops:
                    if i1 == pop1 and i3 == pop1:  # pi2_pop1_new_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop1 and i3 == pop2:  # pi2_pop1_new_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop1:  # pi2_pop1_new_non-source_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop2 and i3 == pop1:  # pi2_pop2_new_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop2 and i3 == pop2:  # pi2_pop2_new_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == pop2:  # pi2_pop2_new_non-source_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i3 == pop1:  # pi2_non_new_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i3 == pop2:  # pi2_non_new_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 2

                    elif i1 == i3:  # pi2_non_new_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (1 - f) ** 2

                    else:  # pi2_non1_new_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            f ** 2
                        )
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f * (1 - f)
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (1 - f) ** 2

                elif i4 == num_pops:
                    if (
                        i1 == pop1 and i2 == pop1 and i3 == pop1
                    ):  # pi2_pop1_pop1_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop1 and i2 == pop1 and i3 == pop2
                    ):  # pi2_pop1_pop1_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i2 == pop1:  # pi2_pop1_pop1_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop1 and i2 == pop2 and i3 == pop1
                    ):  # pi2_pop1_pop2_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop1 and i2 == pop2 and i3 == pop2
                    ):  # pi2_pop1_pop2_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i2 == pop2:  # pi2_pop1_pop2_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i3 == pop1:  # pi2_pop1_non_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i3 == pop2:  # pi2_pop1_non_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i2 == i3:  # pi2_pop1_non_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{2}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{2}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1:  # pi2_pop1_non1_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{3}".format(pop1, pop2, i2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{3}".format(pop1, pop2, i2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i2 == pop1 and i3 == pop1
                    ):  # pi2_pop2_pop1_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i2 == pop1 and i3 == pop2
                    ):  # pi2_pop2_pop1_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i2 == pop1:  # pi2_pop2_pop1_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i2 == pop2 and i3 == pop1
                    ):  # pi2_pop2_pop2_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i2 == pop2 and i3 == pop2
                    ):  # pi2_pop2_pop2_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i2 == pop2:  # pi2_pop2_pop2_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i3 == pop1:  # pi2_pop2_non_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i3 == pop2:  # pi2_pop2_non_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i2 == i3:  # pi2_pop2_non_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{2}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{2}".format(pop1, pop2, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2:  # pi2_pop2_non1_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{3}".format(pop1, pop2, i2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{3}".format(pop1, pop2, i2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop1 and i3 == pop1:  # pi2_non_pop1_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop1 and i3 == pop2:  # pi2_non_pop1_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop1 and i1 == i3:  # pi2_non_pop1_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop1:  # pi2_non1_pop1_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop2 and i3 == pop1:  # pi2_non_pop2_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop2 and i3 == pop2:  # pi2_non_pop2_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop2 and i1 == i3:  # pi2_non_pop2_non_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == pop2:  # pi2_non1_pop2_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop1 and i1 == i2:  # pi2_non1_non1_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop2 and i1 == i2:  # pi2_non1_non1_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i2 == i3:  # pi2_non1_non1_non1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i2:  # pi2_non1_non1_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop1:  # pi2_non1_non2_pop1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{0}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{1}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop2:  # pi2_non1_non2_pop2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{1}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{1}_{1}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i3:  # pi2_non1_non2_non1_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{2}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{1}_{2}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i2 == i3:  # pi2_non1_non2_non2_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{3}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{1}_{3}".format(pop1, pop2, i1, i2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    else:  # pi2_non1_non2_non3_new
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{0}_{4}".format(pop1, pop2, i1, i2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{2}_{3}_{1}_{4}".format(pop1, pop2, i1, i2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                elif i2 == num_pops:
                    if (
                        i1 == pop1 and i3 == pop1 and i4 == pop1
                    ):  # pi2_pop1_new_pop1_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop1 and i3 == pop1 and i4 == pop2
                    ):  # pi2_pop1_new_pop1_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i3 == pop1:  # pi2_pop1_new_pop1_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop1 and i3 == pop2 and i4 == pop1
                    ):  # pi2_pop1_new_pop2_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop1 and i3 == pop2 and i4 == pop2
                    ):  # pi2_pop1_new_pop2_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i3 == pop2:  # pi2_pop1_new_pop2_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i4 == pop1:  # pi2_pop1_new_non_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i4 == pop2:  # pi2_pop1_new_non_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1 and i3 == i4:  # pi2_pop1_new_non_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{2}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{2}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop1:  # pi2_pop1_new_non1_non2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{0}_{2}_{3}".format(pop1, pop2, i3, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{2}_{3}".format(pop1, pop2, i3, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i3 == pop1 and i4 == pop1
                    ):  # pi2_pop2_new_pop1_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{0}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i3 == pop1 and i4 == pop2
                    ):  # pi2_pop2_new_pop1_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i3 == pop1:  # pi2_pop2_new_pop1_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i3 == pop2 and i4 == pop1
                    ):  # pi2_pop2_new_pop2_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif (
                        i1 == pop2 and i3 == pop2 and i4 == pop2
                    ):  # pi2_pop2_new_pop2_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{1}".format(pop1, pop2)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i3 == pop2:  # pi2_pop2_new_pop2_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{2}".format(pop1, pop2, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i4 == pop1:  # pi2_pop2_new_non_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{0}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i4 == pop2:  # pi2_pop2_new_non_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{1}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2 and i3 == i4:  # pi2_pop2_new_non_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{2}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{2}_{2}".format(pop1, pop2, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == pop2:  # pi2_pop2_new_non1_non2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{1}_{2}_{3}".format(pop1, pop2, i3, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{1}_{2}_{3}".format(pop1, pop2, i3, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop1 and i4 == pop1:  # pi2_non_new_pop1_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{0}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop1 and i4 == pop2:  # pi2_non_new_pop1_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop1 and i1 == i4:  # pi2_non_new_pop1_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop1:  # pi2_non1_new_pop1_non2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{3}".format(pop1, pop2, i1, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{3}".format(pop1, pop2, i1, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop2 and i4 == pop1:  # pi2_non_new_pop2_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop2 and i4 == pop2:  # pi2_non_new_pop2_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{1}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop2 and i1 == i4:  # pi2_non_new_pop2_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i3 == pop2:  # pi2_non1_new_pop2_non2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{3}".format(pop1, pop2, i1, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{3}".format(pop1, pop2, i1, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i3 and i4 == pop1:  # pi2_non_new_non_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i3 and i4 == pop2:  # pi2_non_new_non_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i3 == i4:  # pi2_non_new_non_non
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{2}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{2}_{2}".format(pop1, pop2, i1)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i1 == i3:  # pi2_non1_new_non1_non2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{2}_{3}".format(pop1, pop2, i1, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{2}_{3}".format(pop1, pop2, i1, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i4 == pop1:  # pi2_non1_new_non2_pop1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{0}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i4 == pop2:  # pi2_non1_new_non2_pop2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{1}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i4 == i1:  # pi2_non1_new_non2_non1
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{2}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{2}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    elif i4 == i3:  # pi2_non1_new_non2_non2
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{3}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{3}_{3}".format(pop1, pop2, i1, i3)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                    else:  # pi2_non1_new_non2_non3
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{0}_{2}_{3}_{4}".format(pop1, pop2, i1, i3, i4)
                                )
                            ),
                        ] += f
                        A[
                            ii,
                            moms_from.index(
                                Util.map_moment(
                                    "pi2_{1}_{2}_{3}_{4}".format(pop1, pop2, i1, i3, i4)
                                )
                            ),
                        ] += (
                            1 - f
                        )

                else:
                    print("missed a pi2 : ", mom_to)
    return A
