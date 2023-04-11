import numpy as np
from scipy.special import gammaln
from scipy.sparse import csc_matrix
from scipy.sparse import identity
import scipy.sparse
import moments.TwoLocus.Numerics
from copy import copy
import os, sys

import pickle


def save_pickle(matrix, filename):
    with open(filename, "wb+") as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as infile:
        infile.seek(0)
        try:
            matrix = pickle.load(infile)
        except:
            matrix = pickle.load(infile, encoding="Latin1")
    return matrix


# Cache jackknife matrices in ~/.moments/TwoLocus_cache by default
def set_cache_path(path="~/.moments/TwoLocus_cache/"):
    """
    Set directory in which demographic equilibrium phi spectra will be cached.

    The collection of cached spectra can get large, so it may be helpful to
    store them outside the user's home directory.
    """
    global cache_path
    cache_path = os.path.expanduser(path)
    try:
        os.makedirs(cache_path)
    except FileExistsError:
        pass


cache_path = None
set_cache_path()

# Jackknife for interior points
# grab closest 10 points that, for n->n+1


def closest_ijk(i, j, k, n, jump):
    fi, fj, fk = i / (n + float(jump)), j / (n + float(jump)), k / (n + float(jump))
    possible_ijk = []
    for ii in range(1, n):
        for jj in range(1, n - ii):
            for kk in range(1, n - ii - jj):
                possible_ijk.append((ii, jj, kk))
    possible_ijk = np.array(possible_ijk)
    ## find a list of >10 (~20 enough) closest points, fill in starting with closest first
    ## add to list of ordered_set with the following conditions:
    ## - range = exactly 2
    ## - no more than 6 in a given column, row, or depth
    smallests = np.argpartition(
        np.sum((np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1),
        min(40, len(possible_ijk)) - 1,
    )[: min(40, len(possible_ijk)) - 1]
    smallest_set = np.array([possible_ijk[l] for l in smallests])
    distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1)
    order = distances.argsort()
    smallest_set = np.array([smallest_set[l] for l in order])
    ii_vals = []
    jj_vals = []
    kk_vals = []
    ordered_set = []
    ind = 0
    while len(ordered_set) < 10:
        iijjkk = smallest_set[ind]
        ii, jj, kk = iijjkk
        # ranges <= 3
        if len(set(ii_vals)) == 3 and ii not in ii_vals:
            ind += 1
            continue
        if len(set(jj_vals)) == 3 and jj not in jj_vals:
            ind += 1
            continue
        if len(set(kk_vals)) == 3 and kk not in kk_vals:
            ind += 1
            continue
        # not more than six per plane
        if ii_vals.count(ii) == 6 or jj_vals.count(jj) == 6 or kk_vals.count(kk) == 6:
            ind += 1
            continue
        # ranges = 3 by the end
        num_unfilled = (
            3 - len(set(ii_vals)) + 3 - len(set(jj_vals)) + 3 - len(set(kk_vals))
        )
        # if we need to expand the range, skip if it doesn't
        if len(ordered_set) >= 10 - num_unfilled:
            if (
                ((len(set(ii_vals)) < 3 and ii in ii_vals) or (len(set(ii_vals)) == 3))
                and (
                    (len(set(jj_vals)) < 3 and jj in jj_vals)
                    or (len(set(jj_vals)) == 3)
                )
                and (
                    (len(set(kk_vals)) < 3 and kk in kk_vals)
                    or (len(set(kk_vals)) == 3)
                )
            ):
                ind += 1
                continue

        ordered_set.append(iijjkk)
        ii_vals.append(iijjkk[0])
        jj_vals.append(iijjkk[1])
        kk_vals.append(iijjkk[2])
        ind += 1

    ordered_set = np.array(ordered_set)
    # check that matrix rank of get_A(ordered_set,n) has full rank
    A = get_A(ordered_set, n, jump)
    while np.linalg.matrix_rank(A) < 10:
        iijjkk = smallest_set[ind]
        ii, jj, kk = iijjkk
        if ii not in set(ii_vals) or jj not in set(jj_vals) or kk not in set(kk_vals):
            ind += 1
            continue
        # replace one that doesn't reduce any range or cause more than six in a plane
        for blah in range(1, 6):  ## this range is arbitrary
            test = copy(ordered_set)
            test[-blah] = iijjkk
            A = get_A(test, n, jump)
            if np.linalg.matrix_rank(A) == 10:
                ordered_set = test
                break
        ind += 1

    return ordered_set


def closest_ijk_sides(i, j, k, n, jump):
    # closest six points along sides (2D)
    fi, fj, fk = i / (n + float(jump)), j / (n + float(jump)), k / (n + float(jump))

    possible_ijk = []
    if i == 0:
        for jj in range(1, n):
            for kk in range(1, n - jj):
                possible_ijk.append((i, jj, kk))
    elif j == 0:
        for ii in range(1, n):
            for kk in range(1, n - ii):
                possible_ijk.append((ii, j, kk))
    elif k == 0:
        for ii in range(1, n):
            for jj in range(1, n - ii):
                possible_ijk.append((ii, jj, k))
    elif n + jump - i - j - k == 0:
        for ii in range(1, n):
            for jj in range(1, n - ii):
                kk = n - ii - jj
                possible_ijk.append((ii, jj, kk))
    else:
        print("not on a side")
    possible_ijk = np.array(possible_ijk)
    smallests = np.argpartition(
        np.sum((np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1), 6
    )[:6]
    smallest_set = np.array([possible_ijk[l] for l in smallests])
    distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1)
    order = distances.argsort()
    ordered_set = np.array([smallest_set[l] for l in order])
    # ensure that we have an index range of three in each direction
    # if we don't, drop the last (farthest) point, and get next closest until we have three points in each direction
    i_range, j_range, k_range = (
        np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
        np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
        np.max(ordered_set[:, 2]) - np.min(ordered_set[:, 2]),
    )
    ## XXXX update this to reflect the code above
    next_index = 7
    if i == 0:
        while j_range < 2 or k_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            j_range, k_range = (
                np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
                np.max(ordered_set[:, 2]) - np.min(ordered_set[:, 2]),
            )
            next_index += 1
    if j == 0:
        while i_range < 2 or k_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            i_range, k_range = (
                np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
                np.max(ordered_set[:, 2]) - np.min(ordered_set[:, 2]),
            )
            next_index += 1
    if k == 0:
        while i_range < 2 or j_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            i_range, j_range = (
                np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
                np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
            )
            next_index += 1
    if n + jump - i - j - k == 0:
        while i_range < 2 or j_range < 2 or k_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            i_range, j_range, k_range = (
                np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
                np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
                np.max(ordered_set[:, 2]) - np.min(ordered_set[:, 2]),
            )
            next_index += 1

    return ordered_set


def closest_ijk_edges(i, j, k, n, jump):
    # closest three points along edges (1D)
    fi, fj, fk = i / (n + float(jump)), j / (n + float(jump)), k / (n + float(jump))

    # only care about AB/ab and Ab/aB lines (or fA,fB = 0)
    # 4/5/19: with reversible mutation model, care about all sites
    possible_ijk = []
    if j == 0 and k == 0:
        for ii in range(1, n):
            possible_ijk.append((ii, j, k))
    elif i == 0 and n + jump - i - j - k == 0:
        for jj in range(1, n):
            kk = n - jj
            possible_ijk.append((i, jj, kk))
    elif i == 0 and j == 0:
        for kk in range(1, n):
            possible_ijk.append((i, j, kk))
    elif i == 0 and k == 0:
        for jj in range(1, n):
            possible_ijk.append((i, jj, k))
    elif j == 0 and n + jump - i - j - k == 0:
        for ii in range(1, n):
            kk = n - ii
            possible_ijk.append((ii, j, kk))
    elif k == 0 and n + jump - i - j - k == 0:
        for ii in range(1, n):
            jj = n - ii
            possible_ijk.append((ii, jj, k))
    else:
        pass

    possible_ijk = np.array(possible_ijk)
    smallests = np.argpartition(
        np.sum((np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1), 3
    )[:3]
    smallest_set = np.array([possible_ijk[l] for l in smallests])
    distances = np.sum((np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1)
    order = distances.argsort()
    ordered_set = np.array([smallest_set[l] for l in order])
    # ensure that we have an index range of three in each direction
    # if we don't, drop the last (farthest) point, and get next closest until we have three points in each direction
    i_range, j_range, k_range = (
        np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0]),
        np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1]),
        np.max(ordered_set[:, 2]) - np.min(ordered_set[:, 2]),
    )
    # XXX don't think this is needed
    next_index = 4
    if i == 0 and j == 0:  # all aB/ab
        while k_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            k_range = np.max(ordered_set[:, 2]) - np.min(ordered_set[:, 2])
            next_index += 1
    elif i == 0 and k == 0:  # Ab/ab
        while j_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            j_range = np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1])
            next_index += 1
    elif j == 0 and k == 0:  # AB/ab
        while i_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            i_range = np.max(ordered_set[:, 0]) - np.min(ordered_set[:, 0])
            next_index += 1
    elif i == 0 and n + jump - i - j - k == 0:  # Ab/aB
        while j_range < 2:
            smallests = np.argpartition(
                np.sum(
                    (np.array([fi, fj, fk]) - possible_ijk / (1.0 * n)) ** 2, axis=1
                ),
                next_index,
            )[:next_index]
            smallest_set = np.array([possible_ijk[l] for l in smallests])
            distances = np.sum(
                (np.array(smallest_set) / float(n) - [fi, fj, fk]) ** 2, axis=1
            )
            order = distances.argsort()
            new_ordered_set = np.array([smallest_set[l] for l in order])
            ordered_set[-1] = new_ordered_set[-1]
            j_range = np.max(ordered_set[:, 1]) - np.min(ordered_set[:, 1])
            next_index += 1

    elif j == 0 and n + jump - i - j - k == 0:  # AB/aB
        while i_range < 2 or k_range < 2:
            print("oh no")

    elif k == 0 and n + jump - i - j - k == 0:  # AB/Ab
        while i_range < 2 or j_range < 2:
            print("oh no")

    return ordered_set


## these don't depend on the n->n+1 vs n+2


def get_A(ordered_set, n, jump):
    A = np.zeros((10, 10))
    A[0] = 1
    A[1] = ordered_set[:, 0] + 1.0
    A[2] = ordered_set[:, 1] + 1.0
    A[3] = ordered_set[:, 2] + 1
    A[4] = (ordered_set[:, 0] + 2.0) * (ordered_set[:, 0] + 1.0)
    A[5] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 1] + 1.0)
    A[6] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 2] + 1.0)
    A[7] = (ordered_set[:, 1] + 2.0) * (ordered_set[:, 1] + 1.0)
    A[8] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 2] + 1.0)
    A[9] = (ordered_set[:, 2] + 2.0) * (ordered_set[:, 2] + 1.0)
    return A


def compute_alphas(i, j, k, ordered_set, n, jump):
    A = get_A(ordered_set, n, jump)
    b = np.zeros(10)
    b[0] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
    )
    b[1] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        * (i + 1.0)
    )
    b[2] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        * (j + 1.0)
    )
    b[3] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        * (k + 1.0)
    )
    b[4] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        * (n + 5.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        / (n + 5.0 + jump)
        * (i + 2.0)
        * (i + 1.0)
    )
    b[5] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        * (n + 5.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        / (n + 5.0 + jump)
        * (i + 1.0)
        * (j + 1.0)
    )
    b[6] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        * (n + 5.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        / (n + 5.0 + jump)
        * (i + 1.0)
        * (k + 1.0)
    )
    b[7] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        * (n + 5.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        / (n + 5.0 + jump)
        * (j + 2.0)
        * (j + 1.0)
    )
    b[8] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        * (n + 5.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        / (n + 5.0 + jump)
        * (j + 1.0)
        * (k + 1.0)
    )
    b[9] = (
        (n + 1.0)
        * (n + 2.0)
        * (n + 3.0)
        * (n + 4.0)
        * (n + 5.0)
        / (n + 1.0 + jump)
        / (n + 2.0 + jump)
        / (n + 3.0 + jump)
        / (n + 4.0 + jump)
        / (n + 5.0 + jump)
        * (k + 2.0)
        * (k + 1.0)
    )
    return np.linalg.solve(A, b)


def compute_alphas_sides(i, j, k, ordered_set, n, jump):
    A = np.zeros((6, 6))
    b = np.zeros(6)
    # depends what surface we are on
    if (ordered_set[:, 0] == 0).all():  # i = 0
        A[0] = 1
        A[1] = ordered_set[:, 1] + 1.0
        A[2] = ordered_set[:, 2] + 1.0
        A[3] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
        A[4] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 2] + 1.0)
        A[5] = (ordered_set[:, 2] + 1.0) * (ordered_set[:, 2] + 2.0)
        b[0] = (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump)
        b[1] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (j + 1.0)
        )
        b[2] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (k + 1.0)
        )
        b[3] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (j + 1.0)
            * (j + 2.0)
        )
        b[4] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (j + 1.0)
            * (k + 1.0)
        )
        b[5] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (k + 1.0)
            * (k + 2.0)
        )
        return np.linalg.solve(A, b)
    elif (ordered_set[:, 1] == 0).all():  # j = 0
        A[0] = 1
        A[1] = ordered_set[:, 0] + 1.0
        A[2] = ordered_set[:, 2] + 1.0
        A[3] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
        A[4] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 2] + 1.0)
        A[5] = (ordered_set[:, 2] + 1.0) * (ordered_set[:, 2] + 2.0)
        b[0] = (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump)
        b[1] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (i + 1.0)
        )
        b[2] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (k + 1.0)
        )
        b[3] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (i + 1.0)
            * (i + 2.0)
        )
        b[4] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (i + 1.0)
            * (k + 1.0)
        )
        b[5] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (k + 1.0)
            * (k + 2.0)
        )
        return np.linalg.solve(A, b)
    elif (ordered_set[:, 2] == 0).all():  # k = 0
        A[0] = 1
        A[1] = ordered_set[:, 0] + 1.0
        A[2] = ordered_set[:, 1] + 1.0
        A[3] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
        A[4] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 1] + 1.0)
        A[5] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
        b[0] = (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump)
        b[1] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (i + 1.0)
        )
        b[2] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (j + 1.0)
        )
        b[3] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (i + 1.0)
            * (i + 2.0)
        )
        b[4] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (i + 1.0)
            * (j + 1.0)
        )
        b[5] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (j + 1.0)
            * (j + 2.0)
        )
        return np.linalg.solve(A, b)
    else:  # on non-axis surface ... what do we do here...
        A[0] = 1
        A[1] = ordered_set[:, 0] + 1.0
        A[2] = ordered_set[:, 1] + 1.0
        A[3] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
        A[4] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 1] + 1.0)
        A[5] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
        b[0] = (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump)
        b[1] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (i + 1.0)
        )
        b[2] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            * (j + 1.0)
        )
        b[3] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (i + 1.0)
            * (i + 2.0)
        )
        b[4] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (i + 1.0)
            * (j + 1.0)
        )
        b[5] = (
            (n + 1.0)
            * (n + 2.0)
            * (n + 3.0)
            * (n + 4.0)
            / (n + 1.0 + jump)
            / (n + 2.0 + jump)
            / (n + 3.0 + jump)
            / (n + 4.0 + jump)
            * (j + 1.0)
            * (j + 2.0)
        )
        return np.linalg.solve(A, b)


def compute_alphas_edges(i, j, k, ordered_set, n, jump):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    # depends what edge we are on
    if i == 0 and j == 0:  # fA = 0, index by k
        A[0] = 1
        A[1] = ordered_set[:, 2] + 1.0
        A[2] = (ordered_set[:, 2] + 1.0) * (ordered_set[:, 2] + 2.0)
        b = np.array(
            [
                (n + 1.0) / (n + 1.0 + jump),
                (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump) * (k + 1),
                (n + 1.0)
                * (n + 2.0)
                * (n + 3.0)
                / (n + 1.0 + jump)
                / (n + 2.0 + jump)
                / (n + 3.0 + jump)
                * (k + 1)
                * (k + 2),
            ]
        )
        return np.linalg.solve(A, b)
    elif i == 0 and k == 0:  # fB = 0, index by j
        A[0] = 1
        A[1] = ordered_set[:, 1] + 1.0
        A[2] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
        b = np.array(
            [
                (n + 1.0) / (n + 1.0 + jump),
                (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump) * (j + 1),
                (n + 1.0)
                * (n + 2.0)
                * (n + 3.0)
                / (n + 1.0 + jump)
                / (n + 2.0 + jump)
                / (n + 3.0 + jump)
                * (j + 1)
                * (j + 2),
            ]
        )
        return np.linalg.solve(A, b)
    elif i != 0:  # AB/ab edge, index by i
        A[0] = 1
        A[1] = ordered_set[:, 0] + 1.0
        A[2] = (ordered_set[:, 0] + 1.0) * (ordered_set[:, 0] + 2.0)
        b = np.array(
            [
                (n + 1.0) / (n + 1.0 + jump),
                (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump) * (i + 1),
                (n + 1.0)
                * (n + 2.0)
                * (n + 3.0)
                / (n + 1.0 + jump)
                / (n + 2.0 + jump)
                / (n + 3.0 + jump)
                * (i + 1)
                * (i + 2),
            ]
        )
        return np.linalg.solve(A, b)
    else:  # we're on Ab/aB edge, index by j
        A[0] = 1
        A[1] = ordered_set[:, 1] + 1.0
        A[2] = (ordered_set[:, 1] + 1.0) * (ordered_set[:, 1] + 2.0)
        b = np.array(
            [
                (n + 1.0) / (n + 1.0 + jump),
                (n + 1.0) * (n + 2.0) / (n + 1.0 + jump) / (n + 2.0 + jump) * (j + 1),
                (n + 1.0)
                * (n + 2.0)
                * (n + 3.0)
                / (n + 1.0 + jump)
                / (n + 2.0 + jump)
                / (n + 3.0 + jump)
                * (j + 1)
                * (j + 2),
            ]
        )
        return np.linalg.solve(A, b)


jks = {}


def calc_jk(n, jump):
    try:
        return jks[(n, jump)]
    except KeyError:
        # check if it's saved in the cache, if it is not, add to jks and return
        jk_name = "jk_{0}_{1}.mtx".format(n, jump)
        if os.path.isfile(os.path.join(cache_path, jk_name)):
            jks[(n, jump)] = load_pickle(os.path.join(cache_path, jk_name))
            return jks[(n, jump)]
        else:
            # print("creating and caching jackknife matrix for {0}, {1}".format(n, jump))
            row = []
            col = []
            data = []

            # jackknife interior points
            for i in range(1, n + jump):
                for j in range(1, n + jump - i):
                    for k in range(1, n + jump - i - j):
                        ordered_set = closest_ijk(i, j, k, n, jump)
                        alphas = compute_alphas(i, j, k, ordered_set, n, jump)
                        index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                        for coord, alpha in zip(ordered_set, alphas):
                            index = moments.TwoLocus.Numerics.index_n(
                                n, coord[0], coord[1], coord[2]
                            )
                            row.append(index1)
                            col.append(index)
                            data.append(alpha)

            # jackknife sides
            i = 0
            for j in range(1, n + jump):
                for k in range(1, n + jump - j):
                    ordered_set = closest_ijk_sides(i, j, k, n, jump)
                    alphas = compute_alphas_sides(i, j, k, ordered_set, n, jump)
                    index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                    for coord, alpha in zip(ordered_set, alphas):
                        index = moments.TwoLocus.Numerics.index_n(
                            n, coord[0], coord[1], coord[2]
                        )
                        row.append(index1)
                        col.append(index)
                        data.append(alpha)

            j = 0
            for i in range(1, n + jump):
                for k in range(1, n + jump - i):
                    ordered_set = closest_ijk_sides(i, j, k, n, jump)
                    alphas = compute_alphas_sides(i, j, k, ordered_set, n, jump)
                    index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                    for coord, alpha in zip(ordered_set, alphas):
                        index = moments.TwoLocus.Numerics.index_n(
                            n, coord[0], coord[1], coord[2]
                        )
                        row.append(index1)
                        col.append(index)
                        data.append(alpha)

            k = 0
            for i in range(1, n + jump):
                for j in range(1, n + jump - i):
                    ordered_set = closest_ijk_sides(i, j, k, n, jump)
                    alphas = compute_alphas_sides(i, j, k, ordered_set, n, jump)
                    index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                    for coord, alpha in zip(ordered_set, alphas):
                        index = moments.TwoLocus.Numerics.index_n(
                            n, coord[0], coord[1], coord[2]
                        )
                        row.append(index1)
                        col.append(index)
                        data.append(alpha)

            for i in range(1, n + jump):
                for j in range(1, n + jump - i):
                    k = n + jump - i - j
                    ordered_set = closest_ijk_sides(i, j, k, n, jump)
                    alphas = compute_alphas_sides(i, j, k, ordered_set, n, jump)
                    index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                    for coord, alpha in zip(ordered_set, alphas):
                        index = moments.TwoLocus.Numerics.index_n(
                            n, coord[0], coord[1], coord[2]
                        )
                        row.append(index1)
                        col.append(index)
                        data.append(alpha)

            # jackknife edges
            # AB/ab edge
            j = 0
            k = 0
            for i in range(1, n + jump):
                ordered_set = closest_ijk_edges(i, j, k, n, jump)
                alphas = compute_alphas_edges(i, j, k, ordered_set, n, jump)
                index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                for coord, alpha in zip(ordered_set, alphas):
                    index = moments.TwoLocus.Numerics.index_n(
                        n, coord[0], coord[1], coord[2]
                    )
                    row.append(index1)
                    col.append(index)
                    data.append(alpha)

            # Ab/aB edge
            i = 0
            for j in range(1, n + jump):
                k = n + jump - i - j
                ordered_set = closest_ijk_edges(i, j, k, n, jump)
                alphas = compute_alphas_edges(i, j, k, ordered_set, n, jump)
                index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                for coord, alpha in zip(ordered_set, alphas):
                    index = moments.TwoLocus.Numerics.index_n(
                        n, coord[0], coord[1], coord[2]
                    )
                    row.append(index1)
                    col.append(index)
                    data.append(alpha)

            # fB == 0
            i = 0
            k = 0
            for j in range(1, n + jump):
                ordered_set = closest_ijk_edges(i, j, k, n, jump)
                alphas = compute_alphas_edges(i, j, k, ordered_set, n, jump)
                index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                for coord, alpha in zip(ordered_set, alphas):
                    index = moments.TwoLocus.Numerics.index_n(
                        n, coord[0], coord[1], coord[2]
                    )
                    row.append(index1)
                    col.append(index)
                    data.append(alpha)

            # fA == 0
            i = 0
            j = 0
            for k in range(1, n + jump):
                ordered_set = closest_ijk_edges(i, j, k, n, jump)
                alphas = compute_alphas_edges(i, j, k, ordered_set, n, jump)
                index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                for coord, alpha in zip(ordered_set, alphas):
                    index = moments.TwoLocus.Numerics.index_n(
                        n, coord[0], coord[1], coord[2]
                    )
                    row.append(index1)
                    col.append(index)
                    data.append(alpha)

            # AB/Ab edge
            k = 0
            for i in range(1, n + jump):
                j = n + jump - i
                ordered_set = closest_ijk_edges(i, j, k, n, jump)
                alphas = compute_alphas_edges(i, j, k, ordered_set, n, jump)
                index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                for coord, alpha in zip(ordered_set, alphas):
                    index = moments.TwoLocus.Numerics.index_n(
                        n, coord[0], coord[1], coord[2]
                    )
                    row.append(index1)
                    col.append(index)
                    data.append(alpha)

            # AB/aB edge
            j = 0
            for i in range(1, n + jump):
                k = n + jump - i
                ordered_set = closest_ijk_edges(i, j, k, n, jump)
                alphas = compute_alphas_edges(i, j, k, ordered_set, n, jump)
                index1 = moments.TwoLocus.Numerics.index_n(n + jump, i, j, k)
                for coord, alpha in zip(ordered_set, alphas):
                    index = moments.TwoLocus.Numerics.index_n(
                        n, coord[0], coord[1], coord[2]
                    )
                    row.append(index1)
                    col.append(index)
                    data.append(alpha)

            size_from = int((n + 1) * (n + 2) * (n + 3) / 6)
            size_to = int((n + 1 + jump) * (n + 2 + jump) * (n + 3 + jump) / 6)
            jks[(n, jump)] = csc_matrix((data, (row, col)), shape=(size_to, size_from))
            save_pickle(jks[(n, jump)], os.path.join(cache_path, jk_name))
            return jks[(n, jump)]
