import demes

def slice(g, t):
    """
    Slice a Demes graph at a given time, return the top portion with times shifted
    to the slice time.

    :param g: The input resolved Demes graph.
    :param t: The time in the past at which to slice the graph.
    """
    if t < 0:
        raise ValueError("Slice time must be positive")
    if t == 0:
        return g
    else:
        # initialize new builder object
        b = demes.Builder.fromdict(g.asdict())
        b_slice = demes.Builder(
            time_units=b.data["time_units"],
        )
        if b.data["time_units"] != "generations":
            b_slice.data["generation_time"] = b.data["generation_time"]
        # add demes with shifted times
        for d in b.data["demes"]:
            if d["start_time"] <= t:
                continue
            else:
                d = _shift_deme_time(d, t)
                b_slice.add_deme(d.pop("name"))
                for k, v in d.items():
                    b_slice.data["demes"][-1][k] = v
        # adjust times of any pulses
        if "pulses" in b.data:
            for p in b.data["pulses"]:
                if p["time"] <= t:
                    continue
                p["time"] -= t
                b_slice.add_pulse(
                    sources=p["sources"],
                    dest=p["dest"],
                    proportions=p["proportions"],
                    time=p["time"],
                )
        # adjust times of any migrations
        if "migrations" in b.data:
            for m in b.data["migrations"]:
                if m["start_time"] <= t:
                    continue
                m["start_time"] -= t
                m["end_time"] = max(0, m["end_time"] - t)
                b_slice.add_migration(
                    rate=m["rate"],
                    source=m["source"],
                    dest=m["dest"],
                    start_time=m["start_time"],
                    end_time=m["end_time"],
                )
        return b_slice.resolve()

def _shift_deme_time(d, t):
    """
    Returns a new deme dict with start time and epoch end times shifted by t.
    We remove epochs that are more recent than t, and cut off those that span t.
    """
    d_shifted = {}
    for k, v in d.items():
        if k == "start_time":
            d_shifted[k] = v - t
        elif k == "epochs":
            d_shifted[k] = []
            for e in v:
                e["end_time"] = max(0, e["end_time"] - t)
                d_shifted[k].append(e)
                if e["end_time"] == 0:
                    break
        else:
            d_shifted[k] = v
    return d_shifted
