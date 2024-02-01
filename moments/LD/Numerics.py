import numpy as np

import copy
import itertools
from scipy.sparse import identity
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as spinv
import pickle

from . import Matrices
from . import Util

##
## Population split and admixture fucntions, used in moments.LD.LDstats class methods.
##


def split_h(h, pop_to_split, num_pops):
    """
    Split the single-locus (heterozygosity) statistics. `pop_to_split`
    specifies the population index, while `num_pops` specifies the current
    number of populations represented by `h`. Note that `h` must be "full",
    meaning that it contains all statistics given in
    `moments.LD.Util.het_names(num_pops)`.
    """
    h_from = h
    if len(h_from) != len(Util.het_names(num_pops)):
        raise ValueError("length of het statistics does not match specified num_pops")
    h_new = np.empty(int((num_pops + 1) * (num_pops + 2) / 2))
    c = 0
    hn = Util.het_names(num_pops)
    for ii in range(num_pops + 1):
        for jj in range(ii, num_pops + 1):
            if jj == num_pops:
                if ii == jj:
                    h_new[c] = h_from[hn.index("H_{0}_{0}".format(pop_to_split))]
                else:
                    if ii <= pop_to_split:
                        h_new[c] = h_from[
                            hn.index("H_{0}_{1}".format(ii, pop_to_split))
                        ]
                    else:
                        h_new[c] = h_from[
                            hn.index("H_{0}_{1}".format(pop_to_split, ii))
                        ]
            else:
                h_new[c] = h_from[hn.index("H_{0}_{1}".format(ii, jj))]

            c += 1

    return h_new


def split_ld(y, pop_to_split, num_pops):
    """
    Split the two-locus (LD) statistics. `pop_to_split` specifies the
    population index, while `num_pops` specifies the current number of
    populations represented by `y`. Note that `y` must be "full", meaning that
    it contains all statistics given in `moments.LD.Util.ld_names(num_pops)`.
    """
    mom_list_from = Util.ld_names(num_pops)
    mom_list_to = Util.ld_names(num_pops + 1)
    if len(y) != len(mom_list_from):
        raise ValueError("length of input data and number of populations do not match")
    y_new = np.ones(len(mom_list_to))
    for ii, mom_to in enumerate(mom_list_to):
        if mom_to in mom_list_from:
            y_new[ii] = y[mom_list_from.index(mom_to)]
        else:
            mom_to_split = mom_to.split("_")
            for jj in range(1, len(mom_to_split)):
                if int(mom_to_split[jj]) == num_pops:
                    mom_to_split[jj] = str(pop_to_split)
            mom_from = "_".join(mom_to_split)
            y_new[ii] = y[mom_list_from.index(Util.map_moment(mom_from))]
    return y_new


def admix_h(h, num_pops, pop1, pop2, f):
    """
    Given heterozygosity statistics `h`, return a new array of single-locus
    statistics after admixture between `pop1` and `pop2`, with proportions `f`
    and `1-f`, respectively. `num_pops` is given as the number of populations
    represented by `h`.
    """
    Ah = Matrices.admix_h(num_pops, pop1, pop2, f)
    h_new = Ah.dot(h)
    return h_new


def admix_ld(ys, num_pops, pop1, pop2, f):
    """
    Given linkage disequilibrium statistics `y`, return a new array of
    two-locus statistics after admixture between `pop1` and `pop2`, with
    proportions `f` and `1-f`, respectively. `num_pops` is given as the number
    of populations represented by `y`.
    """
    y_new = []
    Ald = Matrices.admix_ld(num_pops, pop1, pop2, f)
    for y in ys:
        y_new.append(Ald.dot(y))
    return y_new


def admix(Y, num_pops, pop1, pop2, f):
    """
    Given a list of LD and heterozygosity statistics, return a new list of
    statistics after admixture between population indexes `pop1` and `pop2`
    with proportions `f` and `1-f`. `num_pops` is given as the number of
    populations represented by statistics in `Y`. The last element in `Y`
    should be the array of single-locus statistics, while the rest are LD
    statistics for different recombination values.
    """
    h = Y[-1]
    h_new = admix_h(h, num_pops, pop1, pop2, f)
    if len(Y) > 1:
        ys = Y[:-1]
        ys_new = admix_ld(ys, num_pops, pop1, pop2, f)
        return ys_new + [h_new]
    else:
        return [h_new]


##
## Transition matrices
##


def drift(num_pops, nus, frozen=None, rho=None):
    """
    Build the heterozygosity and LD drift transition matrices, for a given
    number of populations and relative population sizes.
    """
    Dh = Matrices.drift_h(num_pops, nus, frozen=frozen)
    if rho is not None:
        Dld = Matrices.drift_ld(num_pops, nus, frozen=frozen)
    else:
        Dld = None
    return Dh, Dld


def mutation(num_pops, theta, frozen=None, selfing=None, rho=None):
    """
    Build the mutation matrices for a given number of populations and theta.
    `rho` is provided to check if we need to build the mutation transition
    matrix for LD statistics. If it is None, we assume we are only working
    with single-locus statistics.
    """
    ### mutation for ld also has dependence on H
    Uh = Matrices.mutation_h(num_pops, theta, frozen=frozen, selfing=selfing)
    if rho is not None:
        Uld = Matrices.mutation_ld(num_pops, theta, frozen=frozen, selfing=selfing)
    else:
        Uld = None
    return Uh, Uld


def recombination(num_pops, rho=0.0, frozen=None, selfing=None):
    """
    Build the recombination transition matrices for the given number of
    populations.
    """
    if np.isscalar(rho):
        R = Matrices.recombination(num_pops, rho, frozen=frozen, selfing=selfing)
    else:
        R = [
            Matrices.recombination(num_pops, r, frozen=frozen, selfing=selfing)
            for r in rho
        ]
    return R


def migration(num_pops, m, frozen=None, rho=None):
    """
    Build the migration transition matrices for the given number of populations
    and migration matrix m.
    """
    Mh = Matrices.migration_h(num_pops, m, frozen=frozen)
    if rho is not None:
        Mld = Matrices.migration_ld(num_pops, m, frozen=frozen)
    else:
        Mld = None
    return Mh, Mld


##
## Integration routines
##


def integrate(
    Y,
    nu,
    T,
    dt=None,
    dt_fac=0.1,
    theta=0.001,
    rho=None,
    m=None,
    num_pops=None,
    selfing=None,
    frozen=None,
):
    if num_pops == None:
        num_pops = len(nu)

    h = Y[-1]
    if len(Y) == 2:
        y = Y[0]
    else:
        ys = Y[:-1]

    if callable(nu):
        nus = nu(0)
    else:
        nus = [float(nu_pop) for nu_pop in nu]

    Uh, Uld = mutation(num_pops, theta, frozen=frozen, selfing=selfing, rho=rho)

    if rho is not None:
        # if rho is a scalar, return single matrix, if rho is a list,
        # returns list of matrices
        R = recombination(num_pops, rho=rho, frozen=frozen, selfing=selfing)

    if num_pops > 1 and m is not None:
        if np.any(np.array(m) != 0):
            Mh, Mld = migration(num_pops, m, frozen=frozen, rho=rho)

    dt = T * dt_fac

    dt_last = dt
    nus_last = nus
    elapsed_t = 0

    while elapsed_t < T:
        if elapsed_t + dt > T:
            dt = T - elapsed_t

        if callable(nu):
            nus = nu(elapsed_t + dt / 2.0)

        # recompute matrices if dt or sizes have changed
        if dt != dt_last or nus != nus_last or elapsed_t == 0:
            Dh, Dld = drift(num_pops, nus, frozen=frozen, rho=rho)
            if num_pops > 1 and m is not None and np.any(np.array(m) != 0):
                Ab_h = Dh + Mh
                if rho is not None:
                    if np.isscalar(rho):
                        Ab_ld = Dld + Mld + R
                    else:
                        Ab_ld = [Dld + Mld + R[i] for i in range(len(rho))]
            else:
                Ab_h = Dh
                if rho is not None:
                    if np.isscalar(rho):
                        Ab_ld = Dld + R
                    else:
                        Ab_ld = [Dld + R[i] for i in range(len(rho))]

            # heterozygosity solver
            Ab1_h = np.eye(Ab_h.shape[0]) + dt / 2.0 * Ab_h
            Ab2_h = np.linalg.inv(np.eye(Ab_h.shape[0]) - dt / 2.0 * Ab_h)
            # ld solvers
            if rho is not None:
                if np.isscalar(rho):
                    Ab1_ld = identity(Ab_ld.shape[0], format="csc") + dt / 2.0 * Ab_ld
                    Ab2_ld = factorized(
                        identity(Ab_ld.shape[0], format="csc") - dt / 2.0 * Ab_ld
                    )
                else:
                    Ab1_ld = [
                        identity(Ab_ld[i].shape[0], format="csc") + dt / 2.0 * Ab_ld[i]
                        for i in range(len(rho))
                    ]
                    Ab2_ld = [
                        factorized(
                            identity(Ab_ld[i].shape[0], format="csc")
                            - dt / 2.0 * Ab_ld[i]
                        )
                        for i in range(len(rho))
                    ]

        # forward
        # ld
        if rho is not None:
            if np.isscalar(rho):
                y = Ab1_ld.dot(y) + dt * Uld.dot(h)
            else:
                ys = [Ab1_ld[i].dot(ys[i]) + dt * Uld.dot(h) for i in range(len(ys))]
        # h
        h = Ab1_h.dot(h) + dt * Uh

        # backward
        # ld
        if rho is not None:
            if np.isscalar(rho):
                y = Ab2_ld(y)
            else:
                ys = [Ab2_ld[i](ys[i]) for i in range(len(ys))]
        # h
        h = Ab2_h.dot(h)

        elapsed_t += dt
        dt_last = copy.copy(dt)
        nus_last = copy.copy(nus)

    Y[-1] = h
    if np.isscalar(rho):
        Y[0] = y
    else:
        Y[:-1] = ys

    return Y


##
## Functions to get the steady state solution for given parameters, including
## migration rates which must connect all populations.
##


def _path_exists(m, i, j):
    """
    Using a breadth-first approach.
    """
    if i == j:
        return True
    (n, _) = np.shape(m)
    visited = [False for _ in range(n)]
    visited[i] = True
    traversal = []
    traversal.insert(0, i)
    while len(traversal) > 0:
        i = traversal.pop()
        for k in range(n):
            if m[i][k] > 0:
                if k == j:
                    return True
                if visited[k] == False:
                    visited[k] = True
                    traversal.insert(0, k)
    return False


def _connected_migration_matrix(m):
    """
    Tests if there is a path connecting any two lineages drawn under a given
    migration matrix. This occurs if there is at least one single-direction
    path between all pairs of populations, or if there is exactly one "sink"
    population, but not more. This ensures that a coalescence will occur in
    finite time given any sampling configuration.

    There is surely a more efficient way to test this.
    """
    n0, n1 = np.shape(m)
    if n0 != n1:
        raise ValueError("badly shaped migration matrix")
    if np.min(m) < 0:
        raise ValueError("migration rates cannot be negative")

    # keep track of connections
    c = np.eye(n0, dtype=int)
    all_connected = True
    for pair in itertools.combinations(range(n0), 2):
        if _path_exists(m, pair[0], pair[1]):
            c[pair[0], pair[1]] = 1
        if _path_exists(m, pair[1], pair[0]):
            c[pair[1], pair[0]] = 1
        if c[pair[0], pair[1]] + c[pair[1], pair[0]] == 0:
            all_connected = False
    if all_connected:
        return True
    else:
        # check if exactly one sink from all populations exists
        if np.sum(np.sum(c, axis=1) == n0):
            return True


def steady_state(nus, m=None, rho=None, theta=0.001, selfing_rate=None):
    """
    Returns the steady state for given relative population sizes, recombination
    and mutation rates, selfing rates, and migration matrix (if more than 1
    populations). Returned as a list of LD and pairwise diversity statistics
    that can be passed to `moments.LD.LDstats()`.

    :param nus: List of relative population sizes. For all population sizes to
        be equal and equal to Ne, specify all population sizes as 1.
    :param m: An nxn migration matrix, where n is the number of populations.
        Migration must allow all sampled lineages to coalesce in finite time,
        so that populations must be connected via migration. This is unused
        when there is a single population given by `nus`.
    :param rho: The population size-scaled recombination rate, or list of rates, 4*Ne*r.
    :param theta: The population size-scaled mutation rate, 4*Ne*u.
    :param selfing_rate: List of selfing rates, with same length as `nus`. If
        not given, we assume selfing rates are 0 in each population.
    """
    if len(nus) == 1:
        m = np.zeros((1, 1))
    if len(np.shape(m)) != 2 or np.shape(m)[0] != np.shape(m)[1]:
        raise ValueError("migration matrix must be square")
    elif np.shape(m)[0] != len(nus):
        raise ValueError("migration matrix must have same number of populations as nus")
    if not _connected_migration_matrix(m):
        raise ValueError("migration matrix must have finite coalescence time")
    n = np.shape(m)[0]

    if rho is not None:
        if np.isscalar(rho) and rho < 0:
            raise ValueError("recombination rate cannot be negative")
        elif not np.isscalar(rho):
            for r in rho:
                if r < 0:
                    raise ValueError("recombination rate cannot be negative")
    if theta <= 0:
        raise ValueError("mutation rate must be positive")

    if len(nus) != n:
        raise ValueError("mismatch number of population sizes")
    if np.any([nu <= 0 for nu in nus]):
        raise ValueError("relative population sizes must be positive")

    if selfing_rate is None:
        selfing_rate = [0 for _ in range(n)]
    elif not hasattr(selfing_rate, "__len__") or len(selfing_rate) != n:
        raise ValueError("selfing rates must be given as list")
    else:
        for f in selfing_rate:
            if f < 0 or f > 1:
                raise ValueError("selfing rates must be between 0 and 1")

    # get the n-population steady state of heterozygosity statistics
    Mh = Matrices.migration_h(n, m)
    Dh = Matrices.drift_h(n, nus)
    Uh = Matrices.mutation_h(n, theta, selfing=selfing_rate)
    h_ss = np.linalg.inv(Mh + Dh).dot(-Uh)

    # get the n-population steady state of LD statistics
    if rho is None:
        return [h_ss]

    def two_pop_ld_ss(nus, m, theta, rho, selfing_rate, h_ss):
        U = Matrices.mutation_ld(n, theta, selfing=selfing_rate)
        R = Matrices.recombination(n, rho, selfing=selfing_rate)
        D = Matrices.drift_ld(n, nus)
        M = Matrices.migration_ld(n, m)
        return factorized(D + R + M)(-U.dot(h_ss))

    if np.isscalar(rho):
        y_ss = two_pop_ld_ss(nus, m, theta, rho, selfing_rate, h_ss)
        return [y_ss, h_ss]

    y_ss = []
    for r in rho:
        y_ss.append(two_pop_ld_ss(nus, m, theta, r, selfing_rate, h_ss))
    return y_ss + [h_ss]
