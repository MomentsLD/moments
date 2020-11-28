# This script contains functions to compute the sample SFS from a demography
# defined using demes. Not ever deme graph will be supported, as moments can
# only handle integrating up to five populations, and cannot include selfing or
# cloning.

from collections import defaultdict
import math
import copy
import numpy as np

import moments


def SFS(g, sampled_demes, sample_sizes, sample_times=None, Ne=None, unsampled_n=4):
    """
    Takes a deme graph and computes the SFS. ``demes`` is a package for
    specifying demographic models in a user-friendly, human-readable YAML
    format. This function automatically parses the demographic description
    and returns a SFS for the specified populations and sample sizes.

    This function is new in version 1.1.0. Future developments will allow for
    inference using ``demes``-based demographic descriptions.

    :param g: A ``demes`` DemeGraph from which to compute the SFS. The DemeGraph
        can either be specified as a YAML file, in which case `g` is a string,
        or as a ``DemeGraph`` object.
    :type g: str or :class:`demes.DemeGraph`
    :param sampled_demes: A list of deme IDs to take samples from. We can repeat
        demes, as long as the sampling of repeated deme IDs occurs at distinct
        times.
    :type sampled_demes: list of strings
    :param sample_sizes: A list of the same length as ``sampled_demes``,
        giving the sample sizes for each sampled deme.
    :type sample_sizes: list of ints
    :param sample_times: If None, assumes all sampling occurs at the end of the
        existence of the sampled deme. If there are
        ancient samples, ``sample_times`` must be a list of same length as
        ``sampled_demes``, giving the sampling times for each sampled
        deme. Sampling times are given in time units of the original deme graph,
        so might not necessarily be generations (e.g. if ``g.time_units`` is years)
    :type sapmle_times: list of floats, optional
    :param Ne: reference population size. If none is given, we use the initial
        size of the root deme.
    :type Ne: float, optional
    :param unsampled_n: The default sample size of unsampled demes, which must be
        greater than or equal to 4.
    :type unsampled_n: int, optional
    :return: A ``moments`` site frequency spectrum, with dimension equal to the
        length of ``sampled_demes``, and shape equal to ``sample_sizes`` plus one
        in each dimension, indexing the allele frequency in each deme from 0
        to n[i], where i is the deme index.
    :rtype: :class:`moments.Spectrum`
    """
    if len(sampled_demes) != len(sample_sizes):
        raise ValueError("sampled_demes and sample_sizes must be same length")
    if sample_times is not None and len(sampled_demes) != len(sample_times):
        raise ValueError("sample_times must have same length as sampled_demes")
    for deme in sampled_demes:
        if deme not in g:
            raise ValueError(f"deme {deme} is not in demography")

    if unsampled_n < 4:
        raise ValueError("unsampled_n must be greater than 3")

    if sample_times is None:
        sample_times = [g[d].end_time for d in sampled_demes]

    # for any ancient samples, we need to add frozen branches
    # with this, all "sample times" are at time 0, and ancient sampled demes are frozen
    if np.any(np.array(sample_times) != 0):
        g, sampled_demes, list_of_frozen_demes = _augment_with_ancient_samples(
            g, sampled_demes, sample_times
        )
        sample_times = [0 for _ in sample_times]
    else:
        list_of_frozen_demes = []

    if g.time_units != "generations":
        g, sample_times = _convert_to_generations(g, sample_times)
    for d, n, t in zip(sampled_demes, sample_sizes, sample_times):
        if n < 4:
            raise ValueError("moments fails with sample sizes less than 4")
        if t < g[d].end_time or t >= g[d].start_time:
            raise ValueError("sample time for {deme} must be within its time span")

    # get the list of demographic events from demes, which is a dictionary with
    # lists of splits, admixtures, mergers, branches, and pulses
    demes_demo_events = g.list_demographic_events()

    # get the dict of events and event times that partition integration epochs, in
    # descending order. events include demographic events, such as splits and
    # mergers and admixtures, as well as changes in population sizes or migration
    # rates that require instantaneous changes in the size function or migration matrix.
    # get the list of demes present in each epoch, as a dictionary with non-overlapping
    # adjoint epoch time intervals
    demo_events, demes_present = _get_demographic_events(
        g, demes_demo_events, sampled_demes
    )

    for epoch, epoch_demes in demes_present.items():
        if len(epoch_demes) > 5:
            raise ValueError(
                f"Moments cannot integrate more than five demes at a time. "
                f"Epoch {epoch} has demes {epoch_demes}."
            )

    # get the list of size functions, migration matrices, and frozen attributes from
    # the deme graph and event times, matching the integration times
    nu_funcs, mig_mats, Ts, frozen_pops = _get_integration_parameters(
        g, demes_present, list_of_frozen_demes, Ne=Ne
    )

    # get the sample sizes within each deme, given sample sizes
    deme_sample_sizes = _get_deme_sample_sizes(
        g,
        demo_events,
        sampled_demes,
        sample_sizes,
        demes_present,
        unsampled_n=unsampled_n,
    )

    # compute the SFS
    fs = _compute_sfs(
        demo_events,
        demes_present,
        deme_sample_sizes,
        nu_funcs,
        mig_mats,
        Ts,
        frozen_pops,
    )

    fs = _reorder_fs(fs, sampled_demes)

    return fs


def _convert_to_generations(g, sample_times):
    """
    Takes a deme graph that is not in time units of generations and converts
    times to generations, using the time units and generation times given.
    """
    if g.time_units == "generations":
        return g, sample_times
    else:
        for ii, sample_time in enumerate(sample_times):
            sample_times[ii] = sample_time / g.generation_time
        g = g.in_generations()
        return g, sample_times


def _augment_with_ancient_samples(g, sampled_demes, sample_times):
    """
    Returns a demography object and new sampled demes where we add
    a branch event for the new sampled deme that is frozen.

    New sampled, frozen demes are labeled "{deme}_sampled_{sample_time}".
    Note that we cannot have multiple ancient sampling events at the same
    time for the same deme (for additional samples at the same time, increase
    the sample size).
    """
    frozen_demes = []
    for ii, (sd, st) in enumerate(zip(sampled_demes, sample_times)):
        if st > 0:
            sd_frozen = sd + f"_sampled_{st}"
            frozen_demes.append(sd_frozen)
            sampled_demes[ii] = sd_frozen
            g.deme(
                id=sd_frozen,
                start_time=st,
                end_time=0,
                initial_size=1,
                ancestors=[sd],
            )
    return g, sampled_demes, frozen_demes


def _get_demographic_events(g, demes_demo_events, sampled_demes):
    """
    Returns demographic events and present demes over each epoch.
    Epochs are divided by any demographic event.
    """
    # first get set of all time dividers, from demographic events, migration
    # rate changes, deme epoch changes
    break_points = set()
    for deme in g.demes:
        for e in deme.epochs:
            break_points.add(e.start_time)
            break_points.add(e.end_time)
    for pulse in g.pulses:
        break_points.add(pulse.time)
    for migration in g.migrations:
        break_points.add(migration.start_time)
        break_points.add(migration.end_time)

    # get demes present for each integration epoch
    integration_times = [
        (start_time, end_time)
        for start_time, end_time in zip(
            sorted(list(break_points))[-1:0:-1], sorted(list(break_points))[-2::-1]
        )
    ]

    # find live demes in each epoch, starting with most ancient
    demes_present = defaultdict(list)
    # add demes as they appear from past to present to end of lists
    deme_start_times = defaultdict(list)
    for deme in g.demes:
        deme_start_times[deme.start_time].append(deme.id)

    if math.inf not in deme_start_times.keys():
        raise ValueError("Root deme must have start time as inf")
    if len(deme_start_times[math.inf]) != 1:
        raise ValueError("Deme graph can only have a single root")

    for start_time in sorted(deme_start_times.keys())[::-1]:
        for deme_id in deme_start_times[start_time]:
            end_time = g[deme_id].end_time
            for interval in integration_times:
                if start_time >= interval[0] and end_time <= interval[1]:
                    demes_present[interval].append(deme_id)

    # dictionary of demographic events (pulses, splits, branches, mergers, and
    # admixtures) it's possible that the order of these events will matter
    # also noting here that there can be ambiguity about order of events, that will
    # change the demography... but there should always be a way to write the demography
    # in an unambiguous manner, using different verbs (e.g., two pulse events at the
    # same time with same dest can be converted to an admixture event, and split the
    # dest deme into two demes)
    demo_events = defaultdict(list)
    for pulse in demes_demo_events["pulses"]:
        event = ("pulse", pulse.source, pulse.dest, pulse.proportion)
        demo_events[pulse.time].append(event)
    for branch in demes_demo_events["branches"]:
        event = ("branch", branch.parent, branch.child)
        demo_events[branch.time].append(event)
    for merge in demes_demo_events["mergers"]:
        event = ("merge", merge.parents, merge.proportions, merge.child)
        demo_events[merge.time].append(event)
    for admix in demes_demo_events["admixtures"]:
        event = ("admix", admix.parents, admix.proportions, admix.child)
        demo_events[admix.time].append(event)
    for split in demes_demo_events["splits"]:
        event = ("split", split.parent, split.children)
        demo_events[split.time].append(event)

    # if there are any unsampled demes that end before present and do not have
    # any descendent demes, we need to add marginalization events.
    for deme_id, succs in g.successors.items():
        if deme_id not in sampled_demes and (
            len(succs) == 0
            or np.all([g[succ].start_time > g[deme_id].end_time for succ in succs])
        ):
            event = ("marginalize", deme_id)
            demo_events[g[deme_id].end_time].append(event)

    return demo_events, demes_present


def _get_integration_parameters(g, demes_present, frozen_list, Ne=None):
    """
    Returns a list of size functions, migration matrices, integration times,
    and frozen attributes.
    """
    nu_funcs = []
    integration_times = []
    migration_matrices = []
    frozen_demes = []

    if Ne is None:
        # get root population and set Ne to root size
        for deme_id, preds in g.predecessors.items():
            if len(preds) == 0:
                root_deme = deme_id
                break
        Ne = g[root_deme].epochs[0].initial_size

    for interval, live_demes in sorted(demes_present.items())[::-1]:
        # get intergration time for interval
        T = (interval[0] - interval[1]) / 2 / Ne
        if T == math.inf:
            T = 0
        integration_times.append(T)
        # get frozen attributes
        freeze = [d in frozen_list for d in live_demes]
        frozen_demes.append(freeze)
        # get nu_function or list of sizes (if all constant)
        sizes = []
        for d in live_demes:
            sizes.append(_sizes_at_time(g, d, interval))
        nu_func = _make_nu_func(sizes, T, Ne)
        nu_funcs.append(nu_func)
        # get migration matrix for interval
        mig_mat = np.zeros((len(live_demes), len(live_demes)))
        for ii, d_from in enumerate(live_demes):
            for jj, d_to in enumerate(live_demes):
                if d_from != d_to:
                    m = _migration_rate_in_interval(g, d_from, d_to, interval)
                    mig_mat[jj, ii] = 2 * Ne * m
        migration_matrices.append(mig_mat)

    return nu_funcs, migration_matrices, integration_times, frozen_demes


def _make_nu_func(sizes, T, Ne):
    """
    Given the sizes at start and end of time interval, and the size function for
    each deme, along with the integration time and reference Ne, return the
    size function that gets passed to the moments integration routines.
    """
    if np.all([s[-1] == "constant" for s in sizes]):
        # all constant
        nu_func = [s[0] / Ne for s in sizes]
    else:
        nu_funcs_separated = []
        for s in sizes:
            if s[-1] == "constant":
                assert s[0] == s[1]
                nu_funcs_separated.append(lambda t, N0=s[0]: N0 / Ne)
            elif s[-1] == "linear":
                nu_funcs_separated.append(
                    lambda t, N0=s[0], NF=s[1]: N0 / Ne + t / T * (NF - N0) / Ne
                )
            elif s[-1] == "exponential":
                nu_funcs_separated.append(
                    lambda t, N0=s[0], NF=s[1]: N0
                    / Ne
                    * np.exp(np.log(NF / N0) * t / T)
                )
            else:
                raise ValueError(f"{s[-1]} not a valid size function")

        def nu_func(t):
            return [nu(t) for nu in nu_funcs_separated]

        # check that this is correct, or if we have to "pin" parameters
    return nu_func


def _sizes_at_time(g, deme_id, time_interval):
    """
    Returns the start size, end size, and size function for given deme over the
    given time interval.
    """
    for epoch in g[deme_id].epochs:
        if epoch.start_time >= time_interval[0] and epoch.end_time <= time_interval[1]:
            break
    if epoch.size_function not in ["constant", "exponential", "linear"]:
        raise ValueError(
            "Can only intergrate constant, exponential, or linear size functions"
        )
    size_function = epoch.size_function

    if size_function == "constant":
        start_size = end_size = epoch.initial_size

    if epoch.start_time == time_interval[0]:
        start_size = epoch.initial_size
    else:
        if size_function == "exponential":
            start_size = epoch.initial_size * np.exp(
                np.log(epoch.final_size / epoch.initial_size)
                * (epoch.start_time - time_interval[0])
                / epoch.time_span
            )
        elif size_function == "linear":
            frac = (epoch.start_time - time_interval[0]) / epoch.time_span
            start_size = epoch.initial_size + frac * (
                epoch.final_size - epoch.initial_size
            )

    if epoch.end_time == time_interval[1]:
        end_size = epoch.final_size
    else:
        if size_function == "exponential":
            end_size = epoch.initial_size * np.exp(
                np.log(epoch.final_size / epoch.initial_size)
                * (epoch.start_time - time_interval[1])
                / epoch.time_span
            )
        elif size_function == "linear":
            frac = (epoch.start_time - time_interval[1]) / epoch.time_span
            end_size = epoch.initial_size + frac * (
                epoch.final_size - epoch.initial_size
            )

    return start_size, end_size, size_function


def _migration_rate_in_interval(g, source, dest, time_interval):
    """
    Get the migration rate from source to dest over the given time interval.
    """
    rate = 0
    for mig in g.migrations:
        if mig.source == source and mig.dest == dest:
            if mig.start_time >= time_interval[0] and mig.end_time <= time_interval[1]:
                rate = mig.rate
    return rate


##
## Functions for SFS computation
##


def _get_deme_sample_sizes(
    g, demo_events, sampled_demes, sample_sizes, demes_present, unsampled_n=4
):
    """
    Returns sample sizes within each deme that is present within each interval.
    Deme samples sizes can change if there are pulse or branching events, e.g.,
    but will be constant over the integration epochs.
    This works by climbing up the demography from most recent integration epoch to
    most distant. Unsampled leaf demes get size unsampled_ns, and others have size
    given by sample_sizes.
    """
    ns = {}
    for interval, deme_ids in demes_present.items():
        ns[interval] = [0 for _ in deme_ids]

    # initialize with sampled demes and unsampled, marginalized demes
    for deme_id, n in zip(sampled_demes, sample_sizes):
        for interval in ns.keys():
            if interval[0] <= g[deme_id].start_time:
                ns[interval][demes_present[interval].index(deme_id)] += n

    # Climb up the demographic events, taking into account pulses, branches, etc
    # when we add a new deme, determine base n from its successors (split, merge,
    # admixture), and propagate up. Similarly, propagate up other events that add
    # lineages to a branch (branches, pulses). Marginalize events add the deme
    # sample size with unsampled_n.
    for t, events in sorted(demo_events.items()):
        for event in events:
            if event[0] == "marginalize":
                deme_id = event[1]
                # add unsampled deme
                for interval in ns.keys():
                    if (
                        interval[0] <= g[deme_id].start_time
                        and interval[1] >= g[deme_id].end_time
                    ):
                        ns[interval][
                            demes_present[interval].index(deme_id)
                        ] += unsampled_n
            elif event[0] == "split":
                # add the parental deme
                deme_id = event[1]
                children = event[2]
                for interval in sorted(ns.keys()):
                    if interval[0] == g[deme_id].end_time:
                        # get child sizes at time of split
                        children_ns = {
                            child: ns[interval][demes_present[interval].index(child)]
                            for child in children
                        }
                    if (
                        interval[0] <= g[deme_id].start_time
                        and interval[1] >= g[deme_id].end_time
                    ):
                        for child in children:
                            ns[interval][
                                demes_present[interval].index(deme_id)
                            ] += children_ns[child]
            elif event[0] == "branch":
                # add child n to parent n for integration epochs above t
                deme_id = event[1]
                child = event[2]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        # get child sizes at time of split
                        child_ns = ns[interval][demes_present[interval].index(child)]
                    if (
                        interval[0] <= g[deme_id].start_time
                        and interval[1] >= g[deme_id].end_time
                        and interval[1] >= t
                    ):
                        ns[interval][demes_present[interval].index(deme_id)] += child_ns
            elif event[0] == "pulse":
                # figure out how much the admix_in_place needs from child to parent
                source = event[1]
                dest = event[2]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        dest_size = ns[interval][demes_present[interval].index(dest)]
                    if (
                        interval[0] <= g[source].start_time
                        and interval[1] >= g[source].end_time
                        and interval[1] >= t
                    ):
                        ns[interval][demes_present[interval].index(source)] += dest_size
            elif event[0] == "merge":
                # each parent gets number of lineages in child
                parents = event[1]
                child = event[3]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        child_size = ns[interval][demes_present[interval].index(child)]
                    for parent in parents:
                        if (
                            interval[0] <= g[parent].start_time
                            and interval[1] >= g[parent].end_time
                        ):
                            ns[interval][
                                demes_present[interval].index(parent)
                            ] += child_size
            elif event[0] == "admix":
                # each parent gets num child lineages for all epochs above t
                parents = event[1]
                child = event[3]
                for interval in sorted(ns.keys()):
                    if interval[0] == t:
                        child_size = ns[interval][demes_present[interval].index(child)]
                    for parent in parents:
                        if (
                            interval[0] <= g[parent].start_time
                            and interval[1] >= g[parent].end_time
                            and interval[1] >= t
                        ):
                            ns[interval][
                                demes_present[interval].index(parent)
                            ] += child_size
    return ns


def _compute_sfs(
    demo_events,
    demes_present,
    deme_sample_sizes,
    nu_funcs,
    migration_matrices,
    integration_times,
    frozen_demes,
    theta=1.0,
    gamma=None,
    h=None,
    reversible=False,
):
    """
    Integrates using moments to find the SFS for given demo events, etc
    """
    if gamma is not None and h is None:
        h = 0.5

    if reversible is True:
        assert type(theta) is list
        assert len(theta) == 2
        # theta is forward and backward rates, as list of length 2
        theta_fd = theta[0]
        theta_bd = theta[1]
        assert theta_fd < 1 and theta_bd < 1
    else:
        # theta is a scalar
        assert type(theta) in [int, float]

    integration_intervals = sorted(list(demes_present.keys()))[::-1]

    # set up initial steady-state 1D SFS for ancestral deme
    n0 = deme_sample_sizes[integration_intervals[0]][0]
    if gamma is None:
        gamma0 = 0.0
    if h is None:
        h0 = 0.5
    if reversible is False:
        fs = theta * moments.LinearSystem_1D.steady_state_1D(n0, gamma=gamma0, h=h0)
    else:
        fs = moments.LinearSystem_1D.steady_state_1D_reversible(
            n0, gamma=gamma0, theta_fd=theta_fd, theta_bd=theta_bd
        )
        if h0 != 0.5:
            raise ValueError("only use h=0.5 for reversible model for now...")
    fs = moments.Spectrum(fs, pop_ids=demes_present[integration_intervals[0]])

    # for each set of demographic events and integration epochs, step through
    # integration, apply events, and then reorder populations to align with demes
    # present in the next integration epoch
    for (T, nu, M, frozen, interval) in zip(
        integration_times,
        nu_funcs,
        migration_matrices,
        frozen_demes,
        integration_intervals,
    ):
        if T > 0:
            if gamma is not None:
                gamma_int = [gamma for _ in frozen]
                h_int = [h for _ in frozen]
            else:
                gamma_int = None
                h_int = None
            if reversible:
                fs.integrate(
                    nu,
                    T,
                    m=M,
                    frozen=frozen,
                    gamma=gamma_int,
                    h=h_int,
                    finite_genome=True,
                    theta_fd=theta_fd,
                    theta_bd=theta_bd,
                )
            else:
                fs.integrate(
                    nu, T, m=M, frozen=frozen, gamma=gamma_int, h=h_int, theta=theta
                )

        events = demo_events[interval[1]]
        for event in events:
            fs = _apply_event(fs, event, interval[1], deme_sample_sizes, demes_present)

        if interval[1] > 0:
            # rearrange to next order of demes
            next_interval = integration_intervals[
                [x[0] for x in integration_intervals].index(interval[1])
            ]
            next_deme_order = demes_present[next_interval]
            assert fs.ndim == len(next_deme_order)
            assert np.all([d in next_deme_order for d in fs.pop_ids])
            fs = _reorder_fs(fs, next_deme_order)

    return fs


def _apply_event(fs, event, t, deme_sample_sizes, demes_present):
    e = event[0]
    if e == "marginalize":
        fs = fs.marginalize([fs.pop_ids.index(event[1])])
    elif e == "split":
        children = event[2]
        if len(children) == 1:
            # "split" into just one population (name change)
            fs.pop_ids[fs.pop_ids.index(event[1])] = children[0]
        else:
            # split into multiple children demes
            if len(children) + len(fs.pop_ids) - 1 > 5:
                raise ValueError("Cannot apply split that creates more than 5 demes")
            # get children deme sizes at time t
            for i, ns in deme_sample_sizes.items():
                if i[0] == t:
                    split_sizes = [
                        deme_sample_sizes[i][demes_present[i].index(c)]
                        for c in children
                    ]
                    break
            split_idx = fs.pop_ids.index(event[1])
            # children[0] is placed in split idx, the rest are at the end
            fs = _split_fs(fs, split_idx, children, split_sizes)
    elif e == "branch":
        # branch is a split, but keep the pop_id of parent
        parent = event[1]
        child = event[2]
        children = [parent, child]
        for i, ns in deme_sample_sizes.items():
            if i[0] == t:
                split_sizes = [
                    deme_sample_sizes[i][demes_present[i].index(c)] for c in children
                ]
                break
        split_idx = fs.pop_ids.index(parent)
        fs = _split_fs(fs, split_idx, children, split_sizes)
    elif e in ["admix", "merge"]:
        # two or more populations merge, based on given proportions
        parents = event[1]
        proportions = event[2]
        child = event[3]
        for i, ns in deme_sample_sizes.items():
            if i[0] == t:
                child_size = deme_sample_sizes[i][demes_present[i].index(child)]
        fs = _admix_fs(fs, parents, proportions, child, child_size)
    elif e == "pulse":
        # admixture from one population to another, with some proportion
        source = event[1]
        dest = event[2]
        proportion = event[3]
        for i, ns in deme_sample_sizes.items():
            if i[0] == t:
                target_sizes = [
                    deme_sample_sizes[i][demes_present[i].index(source)],
                    deme_sample_sizes[i][demes_present[i].index(dest)],
                ]
        fs = _pulse_fs(fs, source, dest, proportion, target_sizes)
    else:
        raise ValueError(f"Haven't implemented methods for event type {e}")
    return fs


def _split_fs(fs, split_idx, children, split_sizes):
    """
    Split the SFS into children with split_sizes, from the deme at split_idx.
    """
    i = 1
    while i < len(children):
        fs = fs.split(
            split_idx,
            split_sizes[0] + sum(split_sizes[i + 1 :]),
            split_sizes[i],
            new_ids=[children[0], children[i]],
        )
        i += 1
    return fs


def _admix_fs(fs, parents, proportions, child, child_size):
    """
    Both merge and admixture events use this function, with the only difference that
    merge events remove the parental demes, while admixture events do not.
    """
    if len(parents) >= 2:
        # admix first two pops
        fA = proportions[0] / (proportions[0] + proportions[1])
        fB = proportions[1] / (proportions[0] + proportions[1])
        assert np.isclose(fA, 1 - fB)
        idxA = fs.pop_ids.index(parents[0])
        idxB = fs.pop_ids.index(parents[1])
        fs = fs.admix(idxA, idxB, child_size, fA, new_id=child)
    if len(parents) >= 3:
        # admix third pop
        fAB = (proportions[0] + proportions[1]) / (
            proportions[0] + proportions[1] + proportions[2]
        )
        fC = proportions[2] / (proportions[0] + proportions[1] + proportions[2])
        assert np.isclose(fAB, 1 - fC)
        idxAB = fs.pop_ids.index(child)  # last pop, was added to end
        idxC = fs.pop_ids.index(parents[2])
        fs = fs.admix(idxAB, idxC, child_size, fAB, new_id=child)
    if len(parents) >= 4:
        # admix 4th pop
        fABC = (proportions[0] + proportions[1] + proportions[2]) / (
            proportions[0] + proportions[1] + proportions[2] + proportions[3]
        )
        fD = proportions[3] / (
            proportions[0] + proportions[1] + proportions[2] + proportions[3]
        )
        assert np.isclose(fABC, 1 - fD)
        idxABC = fs.pop_ids.index(child)
        idxD = fs.pop_ids.index(parents[3])
        fs = fs.admix(idxABC, idxC, child_size, fABC, new_id=child)
    if len(parents) == 5:
        # admix 5th pop
        fABCD = (proportions[0] + proportions[1] + proportions[2] + proportions[3]) / (
            proportions[0]
            + proportions[1]
            + proportions[2]
            + proportions[3]
            + proportions[4]
        )
        fE = proportions[4] / (
            proportions[0]
            + proportions[1]
            + proportions[2]
            + proportions[3]
            + proportions[4]
        )
        assert np.isclose(fABCD, 1 - fE)
        idxABCD = fs.pop_ids.index(child)
        idxE = fs.pop_ids.index(parents[4])
        fs = fs.admix(idxABCD, idxE, child_size, fABCD, new_id=child)
    return fs


def _pulse_fs(fs, source, dest, proportion, target_sizes):
    # uses admix in place
    source_idx = fs.pop_ids.index(source)
    dest_idx = fs.pop_ids.index(dest)
    fs = fs.pulse_migrate(source_idx, dest_idx, target_sizes[0], proportion)

    assert fs.sample_sizes[source_idx] == target_sizes[0]
    assert fs.sample_sizes[dest_idx] == target_sizes[1]

    return fs


def _reorder_fs(fs, next_deme_order):
    """
    Takes a SFS with given population order and returns a SFS with the order
    from ``next_deme_order``. Uses ``fs.swap_axes(idx1, idx2)`` sequentially
    through populations.

    :param next_deme_order: List of population IDs in the order of output SFS.
    """
    if np.any([id not in fs.pop_ids for id in next_deme_order]) or np.any(
        [id not in next_deme_order for id in fs.pop_ids]
    ):
        raise ValueError("fs.pop_ids and next_deme_order have mismatched IDs")

    out = copy.deepcopy(fs)
    for ii, swap_id in enumerate(next_deme_order):
        pop_id = out.pop_ids[ii]
        if pop_id != swap_id:
            swap_index = out.pop_ids.index(swap_id)
            out = out.swap_axes(ii, swap_index)
    return out
