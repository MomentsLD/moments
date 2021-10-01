# Infer a demographic history using a demes graph as input.  The input demes
# YAML stores in initial parameter guesses and fixed parameters.  A second YAML
# specifies parameters to be fit that align with the input YAML demography.

try:
    import demes
    import ruamel

    _imported_demes = True
except ImportError:
    _imported_demes = False


def _check_demes_imported():
    if not _imported_demes:
        raise ImportError(
            "To simulate using demes, it must be installed -- "
            "try `pip install demes`"
        )


import moments
import numpy as np
import scipy.optimize
import sys, os


def _get_demes_dict(fname):
    """
    The loaded builder has demes, migrations, and pulses, each as a list of items.
    """
    builder = demes.load_asdict(fname)
    return builder


def _get_params_dict(fname):
    """
    Options:
    - data (optional)
    - uL (default to None)
    - perturb (default to 0)
    - verbose (default to 0)
    - parameters (what to fit)
    - constraints
    """
    with open(fname, "r") as fin:
        options = ruamel.yaml.load(fin, Loader=ruamel.yaml.Loader)
    if "parameters" not in options.keys():
        raise ValueError("parameters to fit must be specified")
    return options


def _get_deme_map(builder):
    deme_map = {deme["name"]: i for i, deme in enumerate(builder["demes"])}
    return deme_map


def _get_value(builder, values):
    """
    If there are more than one, check that they are equal.
    """
    inputs = []
    deme_map = _get_deme_map(builder)
    for value in values:
        if "demes" in value.keys():
            for deme in value["demes"].keys():
                if deme not in deme_map:
                    raise ValueError(
                        f"deme {deme} not in deme graph, "
                        "which has {[d['name'] for d in builder['demes']]}"
                    )
                for k1 in value["demes"][deme].keys():
                    if k1 == "epochs":
                        for k2, attribute in value["demes"][deme][k1].items():
                            try:
                                inputs.append(
                                    builder["demes"][deme_map[deme]][k1][k2][attribute]
                                )
                            except:
                                raise ValueError(
                                    f"can't get {attribute} from epoch {k2} "
                                    f"from deme {deme}"
                                )
                    elif k1 == "start_time":
                        try:
                            inputs.append(builder["demes"][deme_map[deme]][k1])
                        except:
                            raise ValueError(f"can't get {k1} from deme {deme}")
                    else:
                        raise ValueError("Cannot optimize {k1} in deme {deme}")
        if "migrations" in value.keys():
            for mig_idx, attribute in value["migrations"].items():
                inputs.append(builder["migrations"][mig_idx][attribute])
        if "pulses" in value.keys():
            for pulse_idx, attribute in value["pulses"].items():
                if attribute in ["time", "proportion"]:
                    inputs.append(builder["pulses"][pulse_idx][attribute])
                else:
                    raise ValueError("Cannot optimize {attribute} from a pulse event")
    unique_inputs = set(inputs)
    if len(unique_inputs) == 0:
        raise ValueError("Didn't find inputs")
    elif len(unique_inputs) > 1:
        raise ValueError(f"Found multiple input values for {values}")
    else:
        return inputs[0]


def _set_value(builder, values, new_val):
    deme_map = _get_deme_map(builder)
    for value in values:
        if "demes" in value.keys():
            for deme in value["demes"].keys():
                if deme not in deme_map:
                    raise ValueError(
                        f"deme {deme} not in deme graph, "
                        "which has {[d['name'] for d in builder['demes']]}"
                    )
                for k1 in value["demes"][deme].keys():
                    if k1 == "epochs":
                        for k2, attribute in value["demes"][deme][k1].items():
                            try:
                                builder["demes"][deme_map[deme]][k1][k2][
                                    attribute
                                ] = new_val
                            except:
                                raise ValueError(
                                    f"can't set {attribute} for epoch {k2} "
                                    f"in deme {deme}"
                                )
                    elif k1 == "start_time":
                        try:
                            builder["demes"][deme_map[deme]][k1] = new_val
                        except:
                            raise ValueError(f"can't set {k1} for deme {deme}")
                    else:
                        raise ValueError("can't set {k1} in deme {deme}")
        if "migrations" in value.keys():
            for mig_idx, attribute in value["migrations"].items():
                builder["migrations"][mig_idx][attribute] = new_val
        if "pulses" in value.keys():
            for pulse_idx, attribute in value["pulses"].items():
                if attribute in ["time", "proportion"]:
                    builder["pulses"][pulse_idx][attribute] = new_val
                else:
                    raise ValueError("can't set new {attribute} in a pulse event")
    return builder


def _update_builder(builder, options, params):
    for new_val, param in zip(params, options["parameters"]):
        builder = _set_value(builder, param["values"], new_val)
    return builder


def _set_up_params_and_bounds(options, builder):
    param_names = []
    p0 = []
    lower_bound = []
    upper_bound = []
    deme_map = _get_deme_map(builder)
    for param in options["parameters"]:
        param_names.append(param["name"])
        p0.append(_get_value(builder, param["values"]))
        if "lower_bound" in param:
            lower_bound.append(param["lower_bound"])
        else:
            lower_bound.append(0)
        if "upper_bound" in param:
            upper_bound.append(param["upper_bound"])
        else:
            upper_bound.append(np.inf)
    return param_names, np.array(p0), np.array(lower_bound), np.array(upper_bound)


def _set_up_constraints(options, param_names):
    if "constraints" in options:
        # check that constraints are valid
        if not np.all(
            [
                v["constraint"] in ["less_than", "greater_than"]
                for v in options["constraints"]
            ]
        ):
            raise ValueError("Constraints must be 'greater_than' or 'less_than'")
        if not np.all([len(v["params"]) == 2 for v in options["constraints"]]):
            raise ValueError("Constraints must be between two parameters")
        params_to_fit = [o["name"] for o in options["parameters"]]
        for v in options["constraints"]:
            if v["params"][0] not in params_to_fit:
                raise ValueError(f"parameter {v['params'][0]} not in parameters to fit")
            if v["params"][1] not in params_to_fit:
                raise ValueError(f"parameter {v['params'][1]} not in parameters to fit")
        # set up constraints function
        constraints = lambda x: np.array(
            [
                x[param_names.index(cons["params"][0])]
                - x[param_names.index(cons["params"][1])]
                if cons["constraint"] == "greater_than"
                else x[param_names.index(cons["params"][1])]
                - x[param_names.index(cons["params"][0])]
                for cons in options["constraints"]
            ]
        )
        return constraints
    else:
        return None


def _perturb_params_constrained(
    p0, fold, lower_bound=None, upper_bound=None, cons=None, reps=100
):
    tries = 0
    conditions_satisfied = False
    while tries < reps and not conditions_satisfied:
        p_guess = p0 * 2 ** (fold * (2 * np.random.random(len(p0)) - 1))
        conditions_satisfied = True
        if lower_bound is not None:
            p_guess = np.maximum(p_guess, 1.01 * lower_bound)
        if upper_bound is not None:
            p_guess = np.minimum(p_guess, 0.99 * upper_bound)
        if cons is not None:
            if np.any(cons(p_guess) < 0):
                conditions_satisfied = False
        tries += 1
        if tries == reps:
            raise ValueError("Failed to set up initial parameters with constraints")
    return p_guess


def _get_root(g):
    for deme_id, preds in g.predecessors().items():
        if len(preds) == 0:
            return deme_id


_out_of_bounds_val = -1e12
_counter = 0


def _object_func(
    params,
    data,
    builder,
    options,
    lower_bound=None,
    upper_bound=None,
    cons=None,
    verbose=0,
    uL=None,
    fit_ancestral_misid=False,
    output_stream=sys.stdout,
):
    # check bounds
    if lower_bound is not None and np.any(params < lower_bound):
        return -_out_of_bounds_val
    if upper_bound is not None and np.any(params > upper_bound):
        return -_out_of_bounds_val
    # check constraints
    if cons is not None and np.any(cons(params) < 0):
        return -_out_of_bounds_val

    global _counter
    _counter += 1

    # update builder
    demo_params = params[:len(params) - fit_ancestral_misid]
    builder = _update_builder(builder, options, demo_params)

    # build graph and compute SFS
    g = demes.Graph.fromdict(builder)
    sampled_demes = data.pop_ids
    sample_sizes = data.sample_sizes
    model = moments.Demes.SFS(g, sampled_demes, sample_sizes)
    if fit_ancestral_misid:
        model = moments.Misc.flip_ancestral_misid(model, params[-1])

    # get log-likelihood
    if uL is not None:
        root = _get_root(g)
        Ne = g[root].epochs[0].start_size
        theta = 4 * Ne * uL
        model *= theta
        LL = moments.Inference.ll(model, data)
    else:
        LL = moments.Inference.ll_multinom(model, data)

    # print outputs if verbose > 0
    if verbose > 0 and _counter % verbose == 0:
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
        output_stream.write("%-8i, %-12g, %s%s" % (_counter, LL, param_str, os.linesep))

    return -LL


def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(np.exp(log_params), *args, **kwargs)


def optimize(
    deme_graph,
    inference_options,
    data,
    maxiter=1000,
    perturb=0,
    verbose=0,
    uL=None,
    log=True,
    method="fmin",
    fit_ancestral_misid=False,
    misid_guess=None,
    output_stream=sys.stdout,
    output=None,
    overwrite=False,
):
    """
    Optimize demography given a deme graph of initial and fixed parameters and 
    inference options that specify which parameters to fix, bounds, contstraints,
    and other options. Note that many of these options can also be specified in the
    input YAML. If they are, the option specified in the options YAML takes
    precedence.

    :param deme_graph: A YAML file in ``demes`` format.
    :param inference_options: See (url) for how to set this up.
    :data: The SFS to fit, which must have pop_ids specified. Can either be a Spectrum
        object or the file path to the stored frequency spectrum. The populations
        in the SFS need to be present (with matching IDs) in the deme graph.
    :param maxiter: The maximum number of iterations to run optimization. Defaults
        to 1000. Note: maxiter does not seem to work with the Powell method! This
        appears to be a bug within scipy.optimize.
    :param perturb: The perturbation amount of the initial parameters. Defaults to
        zero, in which case we do not perturb the initial parameters in the input
        demes YAML. If perturb is greater than zero, the initial parameters in the input
        YAML are randomly perturbed by up to `perturb`-fold.
    :param verbose: The frequency to print updates to `output_stream` if greater than 1.
    :param uL: The mutuation rate scaled by number of callable sites, so that theta
        is the compound parameter 4*Ne*uL. If given, we optimize with `multinom=False`,
        and theta is determined by taking Ne as the ancestral or root population size,
        which can be fit. Defaults to None. If uL is not given, we use `multinom=True`
        and we likely do not want to fit the ancestral population size.
    :param log: If True, optimize over log of the parameters.
    :param method: The optimization method. Available methods are "fmin", "powell",
        and "lbfgsb". Defaults to "fmin".
    :param fit_ancestral_misid: If True, we fit the probability that the ancestral
        state of a given SNP is misidenitified, resulting in ancestral/derived
        labels being flipped. Note: this is only allowed with *unfolded* spectra.
        Defaults to False.
    :param misid_guess: Defaults to 0.01.
    :output_stream: Defaults to standard output. Can be given an open file stream
        instead or other output stream.
    :param output: If given, the filename for the output best-fit model YAML.
    :param overwrite: If True, overwrites any existing file with the same output
        name.
    """
    _check_demes_imported()
    # load file, data,
    builder = _get_demes_dict(deme_graph)
    options = _get_params_dict(inference_options)

    if isinstance(data, str):
        data = moments.Spectrum.from_file(data)

    if data.pop_ids is None:
        raise ValueError("data SFS must have population IDs given")
    if len(data.pop_ids) != len(data.sample_sizes):
        raise ValueError("pop_ids and sample_sizes have different lengths")

    param_names, p0, lower_bound, upper_bound = _set_up_params_and_bounds(
        options, builder
    )
    cons = _set_up_constraints(options, param_names)

    if fit_ancestral_misid:
        if misid_guess is None:
            misid_guess = 0.02
        param_names.append("p_misid")
        p0 = np.concatenate((p0, [misid_guess]))
        lower_bound = np.concatenate((lower_bound, [0]))
        upper_bound = np.concatenate((upper_bound, [1]))

    # set up extra inputs
    # make sure p0 satisfies constraints and perturb if needed
    if "perturb" in options:
        perturb = options["perturb"]
    if not (isinstance(perturb, float) or isinstance(perturb, int)):
        raise ValueError("perturb must be a non-negative number")
    if perturb < 0:
        raise ValueError("perturb must be non-negative")
    elif perturb > 0:
        p0 = _perturb_params_constrained(p0, perturb, lower_bound, upper_bound, cons)

    # set other input options
    if "uL" in options:
        uL = options["uL"]
    if "verbose" in options:
        verbose = options["verbose"]

    # default is to optimize in log of parameters
    if "log" in options:
        log = options["log"]
        assert log in [True, False]

    # determine method to use
    if "method" in options:
        method = "fmin"
    available_methods = ["fmin", "powell", "lbfgsb"]
    if method not in available_methods:
        raise ValueError(
            f"method {method} not available,  must be one of " f"{available_methods}"
        )

    # set max iterations
    if "maxiter" in options:
        maxiter = options["maxiter"]

    # rescale if log and not lbfgsb
    if log and method != "lbfgsb":
        p0 = np.log(p0)
        obj_fun = _object_func_log
    else:
        obj_fun = _object_func

    args = (
        data,
        builder,
        options,
        lower_bound,
        upper_bound,
        cons,
        verbose,
        uL,
        fit_ancestral_misid,
        output_stream,
    )

    # run optimization
    if method == "fmin":
        outputs = scipy.optimize.fmin(
            obj_fun,
            p0,
            args=args,
            disp=False,
            maxiter=maxiter,
            maxfun=maxiter,
            full_output=True,
        )
        xopt, fopt, iter, funcalls, warnflag = outputs
    elif method == "powell":
        outputs = scipy.optimize.fmin_powell(
            obj_fun,
            p0,
            args=args,
            disp=False,
            maxiter=maxiter,
            maxfun=maxiter,
            full_output=True,
        )
        xopt, fopt, direc, iter, funcalls, warnflag = outputs
    elif method == "lbfgsb":
        bounds = list(zip(lower_bound, upper_bound))
        outputs = scipy.optimize.fmin_l_bfgs_b(
            obj_fun,
            p0,
            bounds=bounds,
            epsilon=1e-2,
            args=args,
            iprint=-1,
            pgtol=1e-5,
            maxiter=maxiter,
            maxfun=maxiter,
            approx_grad=True,
        )
        xopt, fopt, info_dict = outputs

    if log and method != "lbfgsb":
        xopt = np.exp(xopt)

    if output is not None:
        builder = _update_builder(builder, options, xopt)
        g = demes.Graph.fromdict(builder)
        if overwrite is False and os.path.isfile(output):
            output_stream.write(
                f"Did not write output YAML, {output} exists. "
                "To overwrite, set overwrite=True." + os.linesep
            )
        else:
            demes.dump(g, output)

    return param_names, xopt, fopt
