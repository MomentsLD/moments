import demes
import copy
import math


def slice(g, t):
    """
    Slice a Demes graph at a given time, return the top portion with times shifted
    to the slice time.

    :param g: The input resolved Demes graph.
    :param t: The time in the past at which to slice the graph.
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
    ..note::
        Don't really like this function name... any suggestions?

    Returns a new demes graph with demography about the given time removed.
    Demes that existed before that time are removed, and demes that overlap
    with that time are given a constant size equal to their size at that time
    extending into the past.

    :param g: The input demes graph object.
    :param t: The time at which to erase preceding demographic events.
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
