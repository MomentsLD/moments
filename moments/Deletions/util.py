# Utility functions for deletions.
import numpy as np
from scipy.special import gammaln


def choose(n, i):
    return np.exp(gammaln(n + 1) - gammaln(n - i + 1) - gammaln(i + 1))


_idx_cache = {}


def get_idx(n, i, j):
    try:
        return _idx_cache[n][(i, j)]
    except KeyError:
        _idx_cache.setdefault(n, {})
        _idx_cache[n] = cache_idx(n)
        return _idx_cache[n][(i, j)]


def cache_idx(n):
    indexes = {}
    c = 0
    for j in range(n + 1):
        for i in range(n + 1 - j):
            indexes[(i, j)] = c
            c += 1
    return indexes


_proj_cache_1d = {}
_proj_cache = {}


def project_1d(data, n_to):
    n_from = len(data) - 1
    try:
        return _proj_cache_1d[(n_from, n_to)].dot(data)
    except KeyError:
        P = np.zeros((n_to + 1, n_from + 1))
        for i in range(n_to + 1):
            for j in range(n_from + 1):
                P[i, j] = choose(n_from - j, n_to - i) * choose(j, i)
        P /= choose(n_from, n_to)
        _proj_cache_1d[(n_from, n_to)] = P
        return P.dot(data)


def project(data, n_to):
    n_from = np.rint((np.sqrt(1 + 8 * len(data)) - 1) / 2 - 1).astype("int")
    try:
        return _proj_cache[(n_from, n_to)].dot(data)
    except KeyError:
        l_to = (n_to + 1) * (n_to + 2) // 2
        P = np.zeros((l_to, len(data)))
        for i_to in range(n_to + 1):
            for j_to in range(n_to + 1 - i_to):
                for i_from in range(n_from + 1):
                    for j_from in range(n_from + 1 - i_from):
                        if i_to > i_from or j_to > j_from:
                            continue
                        P[
                            get_idx(n_to, i_to, j_to), get_idx(n_from, i_from, j_from)
                        ] = (
                            choose(i_from, i_to)
                            * choose(j_from, j_to)
                            * choose(n_from - i_from - j_from, n_to - i_to - j_to)
                        )
        P /= choose(n_from, n_to)
        _proj_cache[(n_from, n_to)] = P
        return P.dot(data)
