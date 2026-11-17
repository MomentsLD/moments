"""
Likelihood ratio tests of nested Demes models, fit with moments.Demes.Inference.

A test compares the best fits of a complex model and of a nested model, each a Demes
graph with its options file. The nesting is read from the two options files, by the
values in the graph that each parameter sets:

- a complex-model parameter whose values no nested parameter sets is fixed in the
  nested model, at its value in the nested graph (for example, a migration kept in
  the nested graph with rate 0);
- a nested parameter that sets the values of several complex-model parameters is
  shared by them (for example, one size for the start and end of an epoch); each
  complex parameter b sharing a nested parameter with a becomes the nested parameter
  log(b / a), or b - a if the parameters can be zero or negative, which is 0 in the
  nested model;
- other complex-model parameters are free in both models.

The nested graph must have the same structure as the complex graph (the same demes,
epochs, migrations and pulses), so that the complex options apply to it.

As for ``moments.Godambe.LRT_adjust``, the functions return the adjustment factor for
the likelihood ratio statistic ``D = 2 * (ll_complex - ll_nested)``, and the p-value
of ``adj * D`` is ``moments.Godambe.sum_chi2_ppf(adj * D, weights)``, with weights for
the nested parameters that are on a boundary (e.g. ``(0.5, 0.5)`` for a migration rate
fixed at 0).
"""

import copy

import numpy as np

import moments
from . import Inference


def _locations(param):
    # where in the model dict the parameter points - can point to more than one value
    return list(dict.fromkeys(Inference._get_value_locations(param.get("values", []))))


def _nesting(complex_options, nested_options):
    """
    Indices of the complex model's parameters that are fixed in the nested model
    (``fixed``), and groups of them that share a nested parameter (``shared``).
    Raises an error if the nested model isn't nested in the complex one.
    """
    cparams = complex_options["parameters"]
    nparams = nested_options["parameters"]
    owner = {loc: j for j, p in enumerate(nparams) for loc in _locations(p)}
    complex_locs = set()
    # for each nested parameter (its index in the nested options), the indices of the
    # complex parameters whose values it sets
    groups = {}
    # indices of the complex parameters whose values no nested parameter sets: these
    # are fixed in the nested model
    fixed = []
    for i, param in enumerate(cparams):
        locs = _locations(param)
        complex_locs.update(locs)
        owners = {owner.get(loc) for loc in locs}
        if owners == {None}:
            fixed.append(i)
        elif None in owners or len(owners) > 1:
            raise ValueError(
                f"complex parameter {param['name']} is fixed in some of its values "
                "and fit in others, or split between nested parameters, so the "
                "models are not nested"
            )
        else:
            groups.setdefault(owners.pop(), []).append(i)
    for param in nparams:
        # raise if a parameter is fit in the nested model but not in the complex model
        extra = [loc for loc in _locations(param) if loc not in complex_locs]
        if extra:
            raise ValueError(
                f"nested parameter {param['name']} sets {extra}, which the complex "
                "model does not fit, so the models are not nested"
            )
    # the groups of two or more complex parameters that one nested parameter sets: these
    # share that parameter in the nested model (a group of one is a parameter that is
    # free in both models, and is not a nested parameter)
    shared = [sorted(g) for g in groups.values() if len(g) > 1]
    if not fixed and not shared:
        raise ValueError("the nested model fixes or shares no parameters")
    return dict(fixed=fixed, shared=shared)


def _values_and_bounds(builder, options):
    # the parameters' values in a graph and their bounds, without checking one
    # against the other (a nested fit's fixed values can be on a bound)
    params = options["parameters"]
    values = [Inference._get_value(builder, p["values"]) for p in params]
    lower = [p.get("lower_bound", 0) for p in params]
    upper = [p.get("upper_bound", np.inf) for p in params]
    return np.array(values, dtype=float), np.array(lower, float), np.array(upper, float)


class _Transform:
    """
    The complex model's parameters p, transformed to the parameters q in which the
    test is computed. LRT_adjust needs each nested parameter to be a single parameter
    of the complex model, with a fixed value in the nested model. A fixed parameter
    already is, so it is unchanged. Shared parameters are not: the nested model sets
    a = b, so each shared parameter b (with the first of its group, a) is replaced by
    log(b / a), or b - a, which is 0 in the nested model. All other parameters are
    unchanged.

    For example, if the complex model has an epoch's start and end sizes (N0, NF) and
    the nested model has one size for both, the transformed parameters are
    (N0, log(NF / N0)), and the nested model is log(NF / N0) = 0. ``transform`` maps
    p to q, and ``untransform`` maps q back to p to evaluate the model.
    """

    def __init__(self, nest, lower, p_nested):
        self.shared = []  # (first index of the group, shared index, "log" or "diff")
        for group in nest["shared"]:
            a = group[0]
            for b in group[1:]:
                positive = lower[a] >= 0 and lower[b] >= 0 and p_nested[a] > 0
                self.shared.append((a, b, "log" if positive else "diff"))
        self.nested_indices = sorted(nest["fixed"] + [b for _, b, _ in self.shared])

    def transform(self, p):
        q = np.array(p, dtype=float)
        for a, b, kind in self.shared:
            q[b] = np.log(p[b] / p[a]) if kind == "log" else p[b] - p[a]
        return q

    def untransform(self, q):
        p = np.array(q, dtype=float)
        for a, b, kind in self.shared:
            p[b] = q[a] * np.exp(q[b]) if kind == "log" else q[a] + q[b]
        return p


def _setup(complex_graph, complex_options, nested_graph, nested_options):
    """The nesting, transformation and parameter values of a test."""
    options = Inference._get_params_dict(complex_options)
    nest = _nesting(options, Inference._get_params_dict(nested_options))
    p_complex, lower, upper = _values_and_bounds(
        Inference._get_demes_dict(complex_graph), options
    )
    builder = Inference._get_demes_dict(nested_graph)
    try:
        p_nested, _, _ = _values_and_bounds(builder, options)
    except ValueError as err:
        raise ValueError(
            "the complex options don't apply to the nested graph; keep its structure "
            "the same as the complex graph (e.g. a migration with rate 0 rather than "
            f"none): {err}"
        ) from err
    transform = _Transform(nest, lower, p_nested)
    # the step for a parameter at zero is eps times its scale: its value in the
    # complex fit, or else its smallest nonzero finite bound, or else 1
    bound_scale = [
        min([abs(b) for b in (lo, hi) if b != 0 and np.isfinite(b)], default=1.0)
        for lo, hi in zip(lower, upper)
    ]
    scale = np.where(p_complex != 0, np.abs(p_complex), bound_scale)
    for a, b, kind in transform.shared:
        scale[b] = 1.0 if kind == "log" else scale[a]
    # bounds of the transformed parameters (log ratios and differences are unbounded)
    lower_q, upper_q = lower.copy(), upper.copy()
    for _, b, _ in transform.shared:
        lower_q[b], upper_q[b] = -np.inf, np.inf
    return dict(
        options=options,
        builder=builder,
        transform=transform,
        p_nested=p_nested,
        scale=scale,
        bounds=(lower_q, upper_q),
    )


def _adjust(s, loglik, extra, data_args, all_boot, boot_args, eps, labels, caller, ret):
    """
    The adjustment factor at the nested fit, from H, J and the mean score over the
    transformed parameters (see ``_Transform``): the complex model's parameters, then
    the ``extra`` parameters, free parameters of the likelihood that are not in the
    options file (e.g. theta), as (value, lower bound, upper bound). Parameters for
    which a central difference would step outside their bounds use one-sided
    differences, away from the bound.
    """
    transform = s["transform"]
    q0 = transform.transform(np.append(s["p_nested"], [e[0] for e in extra]))
    scale = np.append(s["scale"], np.ones(len(extra)))
    lower = np.append(s["bounds"][0], [e[1] for e in extra])
    upper = np.append(s["bounds"][1], [e[2] for e in extra])
    steps, _ = moments.Godambe._stencils(q0, eps, scale=scale)
    side = np.where(q0 - steps < lower, 1, np.where(q0 + steps > upper, -1, 0))

    def loglik_q(q, *args):
        return loglik(transform.untransform(q), *args)

    _, H, J, cU = moments.Godambe._godambe_matrices(
        loglik_q, q0, data_args, all_boot, boot_args, eps, side=side, scale=scale
    )
    moments.Godambe._check_nested_hessian(H, labels, caller)
    nested_indices = transform.nested_indices
    sub = np.ix_(nested_indices, nested_indices)
    adj = len(nested_indices) / np.trace(np.dot(J[sub], np.linalg.inv(H[sub])))
    if ret:
        return adj, H, J, cU, nested_indices
    return adj


def LRT_adjust(
    complex_graph,
    complex_options,
    nested_graph,
    nested_options,
    data,
    bootstraps,
    uL=None,
    bootstraps_uL=None,
    eps=0.01,
    fit_ancestral_misid=False,
    misid_fit=None,
    return_matrices=False,
):
    """
    Adjustment factor for the likelihood ratio test of a nested Demes model against a
    complex one, both fit to an SFS with ``moments.Demes.Inference.optimize``. It is
    the adjustment of ``moments.Godambe.LRT_adjust``, for the parameters that the
    nested model fixes or shares (see the module documentation).

    :param complex_graph: The complex model's best-fit graph (YAML file).
    :param complex_options: The complex model's options file.
    :param nested_graph: The nested model's best-fit graph, with the same structure
        as the complex graph.
    :param nested_options: The nested model's options file.
    :param data: The data SFS, with ``pop_ids``.
    :param bootstraps: Bootstrap replicate spectra.
    :param uL: The sequence-length-scaled mutation rate, if the fits used one. If
        None, theta is fit by multinomial scaling, as an extra parameter.
    :param bootstraps_uL: Per-bootstrap uL, if uL is given and varies.
    :param eps: Relative step size for finite-difference derivatives.
    :param fit_ancestral_misid: If the ancestral misidentification probability was
        fit, True (as for ``uncerts``).
    :param misid_fit: The nested fit's misidentification probability, if it was fit.
    :param return_matrices: If True, also return H, J and the mean score over all
        parameters (transformed: shared parameters as log ratios or differences,
        then misidentification and theta), and the positions of the nested
        parameters in them.
    :return: The adjustment factor, or ``(adj, H, J, cU, nested_indices)``.
    """
    if fit_ancestral_misid and misid_fit is None:
        raise ValueError("misid_fit must be given if fit_ancestral_misid is True")
    s = _setup(complex_graph, complex_options, nested_graph, nested_options)
    builder, options = copy.deepcopy(s["builder"]), s["options"]
    k = len(s["p_nested"])
    ns = data.sample_sizes
    cache = {}

    def expected(params):
        # the SFS for theta = 1, with ancestral misidentification if fit
        key = tuple(params)
        if key not in cache:
            nonlocal builder
            builder, g = Inference._build_graph_for_uncerts(
                builder, options, params[:k]
            )
            sampled, times = Inference._get_sampled_demes_and_times(data.pop_ids, g)
            fs = moments.Demes.SFS(g, sampled, ns, sample_times=times, u=1)
            if fit_ancestral_misid:
                fs = moments.Misc.flip_ancestral_misid(fs, params[k])
            cache[key] = fs
        return cache[key]

    # extra: free parameters of the likelihood that are not parameters of the Demes
    # model (not in the options file), as (value at the nested fit, lower bound,
    # upper bound): the misidentification probability, if fit, and theta, if uL is
    # not given. They follow the complex model's parameters, and are free in both
    # models.
    extra = [(misid_fit, 0, 1)] if fit_ancestral_misid else []
    if uL is None:
        p0 = list(s["p_nested"]) + [e[0] for e in extra]
        theta = moments.Inference.optimal_sfs_scaling(expected(p0), data)
        extra.append((theta, 0, np.inf))

    def loglik(params, fs, boot_uL):
        n = k + fit_ancestral_misid
        theta = params[n] if uL is None else boot_uL
        return moments.Inference.ll(theta * expected(params[:n]), fs)

    if bootstraps_uL is None:
        bootstraps_uL = [uL] * len(bootstraps)
    labels = [p["name"] for p in options["parameters"]]
    labels += ["p_misid"] * fit_ancestral_misid + ["theta"] * (uL is None)
    return _adjust(
        s,
        loglik,
        extra,
        [data, uL],
        list(zip(bootstraps, bootstraps_uL)),
        lambda b: [moments.Spectrum(b[0]), b[1]],
        eps,
        labels,
        "moments.Demes.LRT.LRT_adjust",
        return_matrices,
    )


def LRT_adjust_LD(
    complex_graph,
    complex_options,
    nested_graph,
    nested_options,
    means,
    varcovs,
    bootstraps,
    pop_ids,
    rs,
    normalization,
    statistics=None,
    eps=0.01,
    return_matrices=False,
):
    """
    Adjustment factor for the likelihood ratio test of a nested Demes model against a
    complex one, both fit to LD statistics with
    ``moments.Demes.Inference.optimize_LD``. Arguments as in ``LRT_adjust`` and
    ``moments.Demes.Inference.uncerts_LD``.

    :return: The adjustment factor, or ``(adj, H, J, cU, nested_indices)``.
    """
    if normalization not in pop_ids:
        raise ValueError("normalization must be one of pop_ids")
    s = _setup(complex_graph, complex_options, nested_graph, nested_options)
    builder, options = copy.deepcopy(s["builder"]), s["options"]
    statistics, means, varcovs, bootstraps = (
        moments.LD.Inference._remove_normalizing_statistics(
            means,
            varcovs,
            bootstraps=bootstraps,
            normalization=pop_ids.index(normalization),
            num_pops=len(pop_ids),
            statistics=statistics,
        )
    )
    cache = {}

    def loglik(params, m):
        key = tuple(params)
        if key not in cache:
            nonlocal builder
            builder, g = Inference._build_graph_for_uncerts(builder, options, params)
            cache[key] = Inference._expected_ld(
                g, pop_ids, rs, normalization, statistics
            )
        return moments.LD.Inference.ll_over_bins(m, cache[key], varcovs)

    return _adjust(
        s,
        loglik,
        [],
        [means],
        bootstraps,
        lambda b: [b],
        eps,
        [p["name"] for p in options["parameters"]],
        "moments.Demes.LRT.LRT_adjust_LD",
        return_matrices,
    )
