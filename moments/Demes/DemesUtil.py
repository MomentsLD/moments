import demes
import copy
import math


def slice(g, t):
    """
    Slice a Demes model at a given time, returning the portion of the
    demographic model above the given time, and all model times are
    shifted to the specified slice time.

    :param g: The input demes model.
    :type g: :class:`demes.Graph`
    :param t: The time in the past at which to slice the graph.
    :type t: scalar
    :return: A new demes model with demography more recent than the
        specified time removed and time shifted to that time ``t``
        is time "zero" in the new model.
    :rtype: :class:`demes.Graph`
    """
    if t < 0:
        raise ValueError("Slice time must be positive")
    if t == math.inf:
        raise ValueError("Slice time must be finite")
    if t == 0:
        return g
    else:
        # initialize new builder object
        gdict = g.asdict()
        b = dict(time_units=gdict["time_units"])
        if gdict["time_units"] != "generations":
            b["generation_time"] = gdict["generation_time"]
        # add demes with shifted times
        b["demes"] = []
        for d in gdict["demes"]:
            if d["start_time"] <= t:
                continue
            else:
                d = _shift_deme_time(d, t)
                b["demes"].append(d)
        # adjust times of any pulses
        if "pulses" in gdict:
            b["pulses"] = []
            for p in gdict["pulses"]:
                if p["time"] <= t:
                    continue
                p["time"] -= t
                b["pulses"].append(p)
        # adjust times of any migrations
        if "migrations" in gdict:
            b["migrations"] = []
            for m in gdict["migrations"]:
                if m["start_time"] <= t:
                    continue
                m["start_time"] -= t
                m["end_time"] = max(0, m["end_time"] - t)
                b["migrations"].append(m)
        b = demes.Builder.fromdict(b)
        return b.resolve()


def _shift_deme_time(d, t):
    """
    Returns a new deme dict with start time and epoch end times shifted by t.
    We remove epochs that are more recent than t, and cut off those that span t.
    """
    d_shifted = {}
    for k, v in d.items():
        if k == "start_time":
            # shift start time
            start_time = v
            d_shifted[k] = v - t
        elif k == "epochs":
            d_shifted[k] = []
            for e in v:
                # get size at slice time to set end size of sliced epoch
                size_at_t = _size_at(
                    t,
                    e["start_size"],
                    e["end_size"],
                    start_time,
                    e["end_time"],
                    e["size_function"],
                )
                start_time = e["end_time"]
                e["end_time"] = max(0, e["end_time"] - t)
                d_shifted[k].append(e)
                if e["end_time"] == 0:
                    # adjust end size
                    d_shifted[k][-1]["end_size"] = size_at_t
                    break
        else:
            d_shifted[k] = v
    return d_shifted


def swipe(g, t):
    """
    Returns a new demes graph with demography above the given time removed.
    Demes that existed before that time are removed, and demes that overlap
    with that time are given a constant size equal to their size at that time
    extending into the past.

    :param g: The input demes model.
    :type g: :class:`demes.Graph`
    :param t: The time at which to erase preceding demographic events.
    :type t: scalar
    :return: A new demes model with demographic events above the specified time
        removed.
    :rtype: :class:`demes.Graph`
    """
    if t <= 0:
        raise ValueError("Slice time must be positive")
    if t == math.inf:
        raise ValueError("Slice time must be finite")

    gdict = g.asdict()
    b = dict(time_units=gdict["time_units"])
    if gdict["time_units"] != "generations":
        b["generation_time"] = gdict["generation_time"]
    b["demes"] = []
    for d in gdict["demes"]:
        if d["epochs"][-1]["end_time"] > t:
            # skip demes that existed entirely before slice time
            continue
        elif d["start_time"] <= t:
            # add demes that started at slice time or more recent
            b["demes"].append(d)
        else:
            # start time is greater than slice time, and end time
            # is equal or more recent than the slice time
            d_new = _get_sliced_deme(d, t)
            b["demes"].append(d_new)
    if "migrations" in gdict:
        b["migrations"] = []
        for m in gdict["migrations"]:
            if m["end_time"] >= t:
                continue
            elif m["start_time"] <= t:
                b["migrations"].append(m)
            else:
                m["start_time"] = t
                b["migrations"].append(m)
    if "pulses" in gdict:
        b["pulses"] = []
        for p in gdict["pulses"]:
            if p["time"] >= t:
                continue
            else:
                b["pulses"].append(p)
    b = demes.Builder.fromdict(b)
    return b.resolve()


def _get_sliced_deme(d, t):
    d_new = dict()
    for k, v in d.items():
        if k in ["ancestors", "proportions"]:
            continue
        elif k == "start_time":
            start_time = v
            d_new["start_time"] = math.inf
        elif k == "epochs":
            d_new["epochs"] = []
            sliced = False
            for e in v:
                if e["end_time"] > t:
                    start_time = e["end_time"]
                    continue
                elif e["end_time"] == t:
                    e["start_size"] = e["end_size"]
                    e["size_function"] = "constant"
                    d_new["epochs"].append(e)
                    sliced = True
                else:
                    if sliced:
                        # already sliced
                        d_new["epochs"].append(e)
                    else:
                        # need to slice
                        s = _size_at(
                            t,
                            e["start_size"],
                            e["end_size"],
                            start_time,
                            e["end_time"],
                            e["size_function"],
                        )
                        e_top = copy.copy(e)
                        e_top["start_size"] = s
                        e_top["end_size"] = s
                        e_top["end_time"] = t
                        e_top["size_function"] = "constant"
                        d_new["epochs"].append(e_top)
                        e_bottom = copy.copy(e)
                        e_bottom["start_size"] = s
                        d_new["epochs"].append(e_bottom)
        else:
            d_new[k] = v
    return d_new


def _size_at(t, start_size, end_size, start_time, end_time, size_function):
    if size_function == "constant":
        assert start_size == end_size
        return start_size
    elif size_function == "exponential":
        T = start_time - end_time
        size_func = lambda tt: start_size * math.exp(
            math.log(end_size / start_size)
            * (start_time - tt)
            / (start_time - end_time)
        )
        return size_func(t)


def rescale(g, Q=1):
    """
    Rescale a demes model by scaling factor ``Q``. This rescaling is done so
    that compound parameters (e.g., ``Ne*m``) remain constant. In the new
    model, population sizes are ``Ne/Q``, model times are `'T/Q'`, and
    migration rates are ``m*Q``.

    For example, setting ``Q=10`` will reduce all population sizes by a factor
    of :math:`1/10`, all times will be reduced by that same amount, and
    migration rates will increase by a factor of 10 so that the product
    ``2*Ne*m`` remains constant.

    When simulating with mutation and recombination rates, or with selection,
    those values should also be scaled by :math:`Q` to ensure that compound
    parameters remain constant.

    Note: As of version 1.2.3, the meaning of ``Q`` has been inverted to
    match standard convention for this scaling parameter.

    :param g: A ``demes`` demographic model.
    :type: :class:`demes.Graph`
    :param Q: The scaling factor. Population sizes and times are scaled by
        dividing values by ``Q``. Migration rates are scaled by multiplying
        by ``Q``. Admixture and ancestry proportions are unchanged, though
        the timing of those events are scaled. Generation times and units
        are unchanged.
    :type: scalar
    :return: A new, rescaled ``demes`` demographic model.
    :rtype: :class:`demes.Graph`
    """
    if Q <= 0 or math.isinf(Q):
        raise ValueError("Scaling factor Q must be positive and finite")
    d = g.asdict()
    for i, deme in enumerate(d["demes"]):
        d["demes"][i]["start_time"] /= Q
        for j, epoch in enumerate(d["demes"][i]["epochs"]):
            d["demes"][i]["epochs"][j]["start_size"] /= Q
            d["demes"][i]["epochs"][j]["end_size"] /= Q
            d["demes"][i]["epochs"][j]["end_time"] /= Q
    for i, mig in enumerate(d["migrations"]):
        d["migrations"][i]["start_time"] /= Q
        d["migrations"][i]["end_time"] /= Q
        d["migrations"][i]["rate"] *= Q
    for i, pulse in enumerate(d["pulses"]):
        d["pulses"][i]["time"] /= Q
    b = demes.Builder.fromdict(d)
    return b.resolve()
