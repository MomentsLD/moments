# Infer a demographic history using a demes graph as input.  The input demes
# YAML stores in initial parameter guesses and fixed parameters.  A second YAML
# specifies parameters to be fit that align with the input YAML demography.

import demes
import ruamel.yaml
import moments
import numpy as np
import scipy.optimize
import sys, os
import warnings
import time


################################################
# Functions shared for SFS and LD optimization #
################################################


def _get_demes_dict(fname):
    """
    The loaded builder has demes, migrations, and pulses, each as a list of items.
    """
    builder = demes.load_asdict(fname)
    return builder


def _get_params_dict(fname):
    """
    Options:
    - parameters (what to fit)
    - constraints
    Below note is from the demes load_dump.py file, re: YAML support in python.
    """
    # NOTE: The state of Python YAML libraries in 2020 leaves much to be desired.
    # The pyyaml library supports only YAML v1.1, which has some awkward corner
    # cases that have been fixed in YAML v1.2. A fork of pyaml, ruamel.yaml,
    # does support YAML v1.2, and introduces a new API for parsing/emitting
    # with additional features and desirable behaviour.
    # However, neither pyyaml nor ruamel guarantee API stability, and neither
    # provide complete reference documentation for their APIs.
    # The YAML code in demes is limited to the following two functions,
    # which are hopefully simple enough to not suffer from API instability.

    # with open(fname, "r") as fin:
    #    options = ruamel.yaml.load(fin, Loader=ruamel.yaml.Loader)
    with ruamel.yaml.YAML(typ="safe") as yaml, open(fname, "r") as fin:
        options = yaml.load(fin)

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
            for deme, k0 in value["demes"].items():
                if deme not in deme_map:
                    raise ValueError(
                        f"deme {deme} not in deme graph, "
                        f"which has {[d['name'] for d in builder['demes']]}"
                    )
                if type(k0) == dict:
                    for k1 in value["demes"][deme].keys():
                        if k1 == "epochs":
                            for k2, attribute in value["demes"][deme][k1].items():
                                try:
                                    inputs.append(
                                        builder["demes"][deme_map[deme]][k1][k2][
                                            attribute
                                        ]
                                    )
                                except:
                                    raise ValueError(
                                        f"can't get {attribute} from epoch {k2} "
                                        f"from deme {deme}"
                                    )
                        elif k1 == "proportions":
                            prop_idx = value["demes"][deme][k1]
                            prop_len = len(builder["demes"][deme_map[deme]][k1])
                            if prop_idx >= prop_len:
                                raise ValueError(
                                    f"can't get proportion index {prop_idx} from deme "
                                    f"{deme}, which has length {prop_len}"
                                )
                            inputs.append(
                                builder["demes"][deme_map[deme]][k1][prop_idx]
                            )
                        else:
                            raise ValueError(
                                f"can't get value from {k1} in deme {deme}"
                            )
                else:
                    if k0 == "start_time":
                        try:
                            inputs.append(builder["demes"][deme_map[deme]][k0])
                        except:
                            raise ValueError(f"can't get {k0} from deme {deme}")
                    else:
                        raise ValueError(f"Cannot optimize {k0} in deme {deme}")
        if "migrations" in value.keys():
            for mig_idx, attribute in value["migrations"].items():
                inputs.append(builder["migrations"][mig_idx][attribute])
        if "pulses" in value.keys():
            for pulse_idx, attribute in value["pulses"].items():
                if attribute == "time":
                    inputs.append(builder["pulses"][pulse_idx][attribute])
                elif type(attribute) == dict:
                    for k, v in attribute.items():
                        inputs.append(builder["pulses"][pulse_idx][k][v])
                else:
                    raise ValueError(f"Cannot optimize {attribute} from a pulse event")
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
            for deme, k0 in value["demes"].items():
                if deme not in deme_map:
                    raise ValueError(
                        f"deme {deme} not in deme graph, "
                        f"which has {[d['name'] for d in builder['demes']]}"
                    )
                if type(k0) == dict:
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
                        elif k1 == "proportions":
                            prop_idx = value["demes"][deme][k1]
                            prop_len = len(builder["demes"][deme_map[deme]][k1])
                            if prop_idx >= prop_len:
                                raise ValueError(
                                    f"can't get proportion index {prop_idx} from deme "
                                    f"{deme}, which has length {prop_len}"
                                )
                            elif prop_idx == prop_len - 1:
                                raise ValueError(
                                    f"can't set last proportion index in deme {deme}"
                                )
                            builder["demes"][deme_map[deme]][k1][prop_idx] = new_val
                            # last entry in proportions must be one minus the sum
                            # of all other entries
                            builder["demes"][deme_map[deme]][k1][-1] = 1 - sum(
                                builder["demes"][deme_map[deme]][k1][:-1]
                            )
                            if np.any(
                                np.array(builder["demes"][deme_map[deme]][k1]) < 0
                            ):
                                raise ValueError(f"negative proportion in deme {deme}")
                            if np.any(
                                np.array(builder["demes"][deme_map[deme]][k1]) > 1
                            ):
                                raise ValueError(
                                    f"proportion larger than 1 in deme {deme}"
                                )
                        else:
                            raise ValueError(
                                f"can't set value from {k1} in deme {deme}"
                            )
                else:
                    if k0 == "start_time":
                        try:
                            builder["demes"][deme_map[deme]][k0] = new_val
                        except:
                            raise ValueError(f"can't set {k0} for deme {deme}")
                    else:
                        raise ValueError(f"can't set {k0} in deme {deme}")
        if "migrations" in value.keys():
            for mig_idx, attribute in value["migrations"].items():
                builder["migrations"][mig_idx][attribute] = new_val
        if "pulses" in value.keys():
            for pulse_idx, attribute in value["pulses"].items():
                if attribute == "time":
                    builder["pulses"][pulse_idx][attribute] = new_val
                elif type(attribute) == dict:
                    for k, v in attribute.items():
                        builder["pulses"][pulse_idx][k][v] = new_val
                else:
                    raise ValueError(f"Cannot set new {attribute} in a pulse event")
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
    p0 = np.array(p0)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    if np.any(lower_bound >= upper_bound):
        raise ValueError("All lower bounds must be less than upper bounds")
    if np.any(p0 <= lower_bound):
        raise ValueError("All initial parameters must be greater than lower bound")
    if np.any(p0 >= upper_bound):
        raise ValueError("All initial parameters must be less than upper bound")
    return param_names, p0, lower_bound, upper_bound


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
    while not conditions_satisfied:
        if tries == reps:
            raise ValueError("Failed to set up initial parameters with constraints")
        # perturb initial parameters and make sure they are within our bounds
        p_guess = p0 * 2 ** (fold * (2 * np.random.random(len(p0)) - 1))
        if np.any(p_guess <= lower_bound) or np.any(p_guess >= upper_bound):
            for i in range(len(p_guess)):
                tries_bounds = 0
                while p_guess[i] <= lower_bound[i] or p_guess[i] >= upper_bound[i]:
                    if tries_bounds == reps:
                        raise ValueError(
                            "Failed to set up initial parameters within bounds"
                        )
                    p_guess[i] = p0[i] * 2 ** (fold * (2 * np.random.random() - 1))
                    tries_bounds += 1
        # check that our constraints are satisfied
        conditions_satisfied = True
        if cons is not None:
            if np.any(cons(p_guess) < 0):
                conditions_satisfied = False
        tries += 1
    return p_guess


def _get_root(g):
    for deme_id, preds in g.predecessors().items():
        if len(preds) == 0:
            return deme_id


def _write_uncerts_output(
    param_names, p0, uncerts_out, output, overwrite, output_stream
):
    output_string = "#param\topt_value\tstd_err\n"
    for param, opt, stderr in zip(param_names, p0, uncerts_out):
        output_string += f"{param}\t{opt}\t{stderr}\n"
    if overwrite is False and os.path.isfile(output):
        output_stream.write(
            f"Did not write output - {output} already exists. The uncertainties "
            "are printed below. To overwrite in future, set overwrite=True."
            + os.linesep
        )
        output_stream.write(output_string)
    else:
        with open(output, "w+") as fout:
            fout.write(output_string)


_out_of_bounds_val = -1e12
_counter = 0


################################################################
# Objective function and inference method for SFS optimization #
################################################################


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
    if cons is not None and np.any(cons(params) <= 0):
        return -_out_of_bounds_val

    global _counter
    _counter += 1

    # update builder
    demo_params = params[: len(params) - fit_ancestral_misid]
    builder = _update_builder(builder, options, demo_params)

    # build graph and compute SFS
    g = demes.Graph.fromdict(builder)
    sampled_demes = data.pop_ids
    sample_sizes = data.sample_sizes

    end_times = {d.name: d.end_time for d in g.demes}
    sample_times = []

    # check for any ancient samples, which have pop id
    # "{deme.name}_sampled_{gen}_{frac_gen}"
    input_sampled_demes = [d for d in sampled_demes]
    for ii, deme in enumerate(sampled_demes):
        if "_sampled_" in deme:
            d = deme.split("_")[0]
            t = float(".".join(deme.split("_")[2:]))
            input_sampled_demes[ii] = d
            sample_times.append(t)
        else:
            sample_times.append(end_times[deme])

    model = moments.Demes.SFS(
        g, input_sampled_demes, sample_sizes, sample_times=sample_times, u=uL
    )
    if fit_ancestral_misid:
        model = moments.Misc.flip_ancestral_misid(model, params[-1])

    # get log-likelihood
    if uL is not None:
        LL = moments.Inference.ll(model, data)
    else:
        LL = moments.Inference.ll_multinom(model, data)

    # print outputs if verbose > 0
    if verbose > 0 and _counter % verbose == 0:
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
        output_stream.write("%-8i, %-12g, %s%s" % (_counter, LL, param_str, os.linesep))
        moments.Misc.delayed_flush(stream=output_stream, delay=0.5)

    return -LL


def _object_func_log(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func(np.exp(log_params - 1), *args, **kwargs)


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
    inference options that specify which parameters to fit, and bounds or constraints
    on those parameters.

    :param deme_graph: A demographic model as a YAML file in ``demes`` format. This
        should be given as a string specifying the path and file name of the model.
    :param inference_options: A second YAML file, specifying the parameters to be
        optimized, parameter bounds, and constraints between parameters. Please see
        the documentation at
        https://momentsld.github.io/moments/extensions/demes.html#the-options-file
    :param data: The SFS to fit, which must have pop_ids specified. Can either be a
        Spectrum object or the file path to the stored frequency spectrum. The
        populations in the SFS (as given by ``sfs.pop_ids``) need to be present in
        the demographic model and have matching IDs.
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
        and "lbfgsb".
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
    :return: List of parameter names, optimal parameters, and LL
    """
    # constraints. Other arguments should be kw args in the function.

    # load file, data,
    builder = _get_demes_dict(deme_graph)
    options = _get_params_dict(inference_options)

    if isinstance(data, str):
        data = moments.Spectrum.from_file(data)

    if data.pop_ids is None:
        raise ValueError("data SFS must specify population IDs")
    if len(data.pop_ids) != len(data.sample_sizes):
        raise ValueError("pop_ids and sample_sizes have different lengths")

    param_names, p0, lower_bound, upper_bound = _set_up_params_and_bounds(
        options, builder
    )
    constraints = _set_up_constraints(options, param_names)

    if fit_ancestral_misid:
        if data.folded:
            raise ValueError(
                "Data is folded - can only fit ancestral misid using unfolded data"
            )
        if misid_guess is None:
            misid_guess = 0.02
        param_names.append("p_misid")
        p0 = np.concatenate((p0, [misid_guess]))
        lower_bound = np.concatenate((lower_bound, [0]))
        upper_bound = np.concatenate((upper_bound, [1]))

    if not (isinstance(perturb, float) or isinstance(perturb, int)):
        raise ValueError("perturb must be a non-negative number")
    if perturb < 0:
        raise ValueError("perturb must be non-negative")
    elif perturb > 0:
        p0 = _perturb_params_constrained(
            p0, perturb, lower_bound, upper_bound, constraints
        )

    available_methods = ["fmin", "powell", "lbfgsb"]
    if method not in available_methods:
        raise ValueError(
            f"method {method} not available,  must be one of {available_methods}"
        )

    # rescale if log
    if log:
        p0 = np.log(p0) + 1
        obj_fun = _object_func_log
    else:
        obj_fun = _object_func

    args = (
        data,
        builder,
        options,
        lower_bound,
        upper_bound,
        constraints,
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
        bounds = list(zip(np.log(lower_bound) + 1, np.log(upper_bound) + 1))
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

    if log:
        xopt = np.exp(xopt - 1)

    if output is not None:
        builder = _update_builder(builder, options, xopt)
        g = demes.Graph.fromdict(builder)
        if overwrite is False and os.path.isfile(output):
            output_stream.write(
                f"Did not write output YAML - {output} already exists. The model is "
                "printed below. To overwrite, set overwrite=True." + os.linesep
            )
            output_stream.write(str(g))
            moments.Misc.delayed_flush(stream=output_stream, delay=0.5)
        else:
            demes.dump(g, output)

    return param_names, xopt, fopt


#################################################################
# Methods for computing confidence intervals from SFS inference #
#################################################################

# Cache evaluations of the frequency spectrum inside our hessian/J
# evaluation function

_sfs_cache = {}


def _get_godambe(
    func_ex,
    all_boot,
    p0,
    data,
    eps,
    uL=None,
    all_boot_uL=None,
    log=False,
    just_hess=False,
):
    # taken and adapted from moments.Godambe, to include bootstraps over uL
    ns = data.sample_sizes

    def func(params, data, uL=None):
        key = (tuple(params), tuple(ns), tuple([uL]))
        if key not in _sfs_cache:
            _sfs_cache[key] = func_ex(params, ns, uL=uL)
        fs = _sfs_cache[key]
        return moments.Inference.ll(fs, data)

    def log_func(logparams, data, uL=None):
        return func(np.exp(logparams), data, uL=uL)

    # First calculate the observed hessian
    if not log:
        hess = -moments.Godambe.get_hess(func, p0, eps, args=[data, uL])
    else:
        hess = -moments.Godambe.get_hess(log_func, np.log(p0), eps, args=[data, uL])

    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = np.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = np.zeros((len(p0), 1))
    if all_boot_uL is None:
        all_boot_uL = [uL for _ in all_boot]
    for ii, (boot, uL) in enumerate(zip(all_boot, all_boot_uL)):
        boot = moments.Spectrum(boot)
        if not log:
            grad_temp = moments.Godambe._get_grad(func, p0, eps, args=[boot, uL])
        else:
            grad_temp = moments.Godambe._get_grad(
                log_func, np.log(p0), eps, args=[boot, uL]
            )

        J_temp = np.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = np.linalg.inv(J)
    godambe = np.dot(np.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


def uncerts(
    deme_graph,
    inference_options,
    data,
    bootstraps=None,
    uL=None,
    bootstraps_uL=None,
    log=False,
    eps=0.01,
    method="FIM",
    fit_ancestral_misid=False,
    misid_fit=None,
    verbose=0,
    output_stream=sys.stdout,
    output=None,
    overwrite=False,
):
    """
    Compute uncertainties for fitted parameters, using the output YAML from
    ``moments.Demes.Inference.optimize()``.

    :param deme_graph: The file path to the output demes graph from `optimize()`.
    :type deme_graph: str
    :param inference_options: The same options file used in the original optimization.
    :type inference_options: str
    :param data: The same data SFS used in the origional optimization.
    :type data: moments.Spectrum
    :param bootstraps: A list of bootstrap replicates of the SFS. See documentation
        for examples of computing block-bootstrap replicate data.
    :type bootstraps: list
    :param uL: The sequence length-scaled mutation rate. This must be provided if
        it was used in the original optimization.
    :type uL: float
    :param bootstraps_uL: If uL was used in the original optimization, this should be
        provided. It is a list of the same length as the bootstrap replicates
        containing the sequence length-scaled mutation rates for the corresponding
        bootstrap SFS replicates, matching the index. If it it not given, uL is assumed
        to be equivalent across bootstrap sets, using the value provided in the uL
        keyword argument.
    :type bootstraps_uL: list
    :param log: Defaults to False. If True, we assume a log-normal distribution of
        parameters. Returned values are then the standard deviations of the *logs*
        of the parameter values, which can be interpreted as relative parameter
        uncertainties.
    :type log: bool
    :param eps: Fractional stepsize to use when taking finite-difference derivatives.
        Note that if eps*param is < 1e-6, then the step size for that parameter
        will simply be eps, to avoid numerical issues with small parameter
        perturbations. Defaults to 0.01.
    :type eps: float
    :param method: Choose between either "FIM" (Fisher Information) or "GIM" (Godambe
        Information). Defaults to FIM. If GIM is chosen, bootstrap replicates must
        be provided.
    :type method: str
    :param fit_ancestral_misid: If we fit the ancestral misidentification in the
        original optimization, set to True.
    :type fit_ancestral_misid: bool
    :param misid_fit: If we fit the ancestral misidentification in the original
        optimization, provide the best fit value here.
    :type misid_fit: float
    :param verbose: If greater than zero, print the progress of iterations needed
        to compute uncertainties. Prints every {verbose} number of iterations.
    :type verbose: int
    :param output_stream: Default is sys.stdout, but can be changed using this
        option.
    :param output: If given, write the output as a tab-delimited table with
        parameter names, best-fit values, and standard errors as columns.
    :type output: str
    :param overwrite: If True, overwrite the output table of uncertainties.
    :type overwrite: bool
    """
    func_calls = 0

    # Get p0 and parameter information
    builder = _get_demes_dict(deme_graph)
    options = _get_params_dict(inference_options)

    if isinstance(data, str):
        data = moments.Spectrum.from_file(data)

    if data.pop_ids is None:
        raise ValueError("data SFS must specify population IDs")
    if len(data.pop_ids) != len(data.sample_sizes):
        raise ValueError("pop_ids and sample_sizes have different lengths")

    param_names, p0, lower_bound, upper_bound = _set_up_params_and_bounds(
        options, builder
    )

    if verbose > 0:
        exp_num_calls = moments.LD.Godambe._expected_number_of_calls(p0)
        output_stream.write(
            "Expected number of function calls: " + str(exp_num_calls) + os.linesep
        )
        moments.Misc.delayed_flush(stream=output_stream, delay=0.5)

    if fit_ancestral_misid:
        if misid_fit is None:
            raise ValueError("misid_fit cannot be None if fit_ancestral_misid is True")
        param_names.append("p_misid")
        p0 = np.concatenate((p0, [misid_fit]))
        lower_bound = np.concatenate((lower_bound, [0]))
        upper_bound = np.concatenate((upper_bound, [1]))

    sample_sizes = data.sample_sizes
    sampled_demes = data.pop_ids

    # Given scaled mutation rate (multinom = False), no extra parameters added
    # But if uL is None (multinom = True), add a scaling parameter (theta)
    if uL is None:
        multinom = True
        g = demes.load(deme_graph)
        model = moments.Demes.SFS(
            g, sampled_demes=sampled_demes, sample_sizes=sample_sizes
        )
        uL = moments.Inference.optimal_sfs_scaling(model, data)
        p0 = list(p0) + [uL]
    else:
        multinom = False

    def func_ex(params, ns, uL=None):
        """
        params is:
        opt_params + [misid_fit] (if fit_ancestral_misid) + [theta] (if uL is None)
        """
        nonlocal builder
        nonlocal options
        nonlocal func_calls
        func_calls += 1

        # pull out uL and p_misid, if needed
        if multinom:
            uL = params[-1]
            demo_params = params[:-1]
        else:
            demo_params = params[:]
        if fit_ancestral_misid:
            p_misid = demo_params[-1]
            demo_params = demo_params[:-1]
        else:
            demo_params = demo_params
            p_misid = 0

        # update the builder dict, then compute SFS
        builder = _update_builder(builder, options, demo_params)
        g = demes.Graph.fromdict(builder)

        sampled_demes = data.pop_ids
        sample_sizes = data.sample_sizes

        end_times = {d.name: d.end_time for d in g.demes}
        sample_times = []

        # check for any ancient samples, which have pop id
        # "{deme.name}_sampled_{gen}_{frac_gen}"
        input_sampled_demes = [d for d in sampled_demes]
        for ii, deme in enumerate(sampled_demes):
            if "_sampled_" in deme:
                d = deme.split("_")[0]
                t = float(".".join(deme.split("_")[2:]))
                input_sampled_demes[ii] = d
                sample_times.append(t)
            else:
                sample_times.append(end_times[deme])

        model = moments.Demes.SFS(
            g, input_sampled_demes, sample_sizes, sample_times=sample_times, u=uL
        )

        if fit_ancestral_misid:
            model = moments.Misc.flip_ancestral_misid(model, p_misid)

        if (verbose > 0) and (func_calls % verbose == 0):
            # param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
            # output_stream.write("%-8i, %s%s" % (func_calls, param_str, os.linesep))
            output_stream.write(
                f"Finished iteration {func_calls: > {len(str(exp_num_calls))}} "
                + f"of {exp_num_calls}"
                + os.linesep
            )
            moments.Misc.delayed_flush(stream=output_stream, delay=0.5)

        return model

    if method == "FIM":
        # computing
        if bootstraps is not None:
            warnings.warn(
                "FIM method chosen but bootstrap replicates provided - "
                "Bootstraps will not be used"
            )
        H = _get_godambe(func_ex, [], p0, data, eps, log=log, uL=uL, just_hess=True)
        uncerts_out = np.sqrt(np.diag(np.linalg.inv(H)))
    elif method == "GIM":
        if bootstraps is None:
            raise ValueError(
                "A list of SFS bootstrap replicates must be provided to use GIM method"
            )
        GIM, H, J, cU = _get_godambe(
            func_ex,
            bootstraps,
            p0,
            data,
            eps,
            log=log,
            uL=uL,
            all_boot_uL=bootstraps_uL,
        )
        uncerts_out = np.sqrt(np.diag(np.linalg.inv(GIM)))
    else:
        raise ValueError("method must be either 'FIM' or 'GIM'")

    if output is not None:
        _write_uncerts_output(
            param_names, p0, uncerts_out, output, overwrite, output_stream
        )

    return uncerts_out


###############################################################
# Objective function and inference method for LD optimization #
###############################################################


def compute_bin_stats(g, sampled_demes, sample_times=None, rs=None):
    """
    Given a list of per-base recombination rates defining recombination bin
    edges, computes expected LD statistics within each bin.

    :param g: A demes-formatted demographic model.
    :param sampled_demes: List of populations to sample.
    :param sample_types: Optional list of sample times for each population.
    :param rs: The list of bin edges, as an array of increasing values. Bins
        are defined using adjacent values, so that if ``rs`` has length n,
        there are n-1 bins.
    :return: An LDstats object.
    """
    # check for valid recombination rates
    if not hasattr(rs, "__len__"):
        raise ValueError("rs must be a list of per-base recombination rates")
    if len(rs) <= 1:
        raise ValueError("rs must have at least two recombination rates")
    for i in range(len(rs) - 1):
        if rs[i + 1] <= rs[i]:
            raise ValueError("rs must be a list of increasing values")
    # rhos are computed internally in Demes.LD() using rs and the root deme initial Ne
    y_edges = moments.Demes.LD(g, sampled_demes, sample_times=sample_times, r=rs)
    r_mids = [(r_l + r_r) / 2 for r_l, r_r in zip(rs[:-1], rs[1:])]
    y_mids = moments.Demes.LD(g, sampled_demes, sample_times=sample_times, r=r_mids)
    # simpson's integration
    y = [
        1 / 6 * (y_edges[i] + y_edges[i + 1] + 4 * y_mids[i])
        for i in range(len(r_mids))
    ]
    y.append(y_edges[-1])
    model = moments.LD.LDstats(y, num_pops=y_edges.num_pops, pop_ids=sampled_demes)
    return model


def _object_func_LD(
    params,
    means,
    varcovs,
    builder,
    options,
    pop_ids=None,
    rs=None,
    statistics=None,
    normalization=None,
    lower_bound=None,
    upper_bound=None,
    cons=None,
    verbose=0,
    output_stream=sys.stdout,
):
    # check bounds
    if lower_bound is not None and np.any(params < lower_bound):
        return -_out_of_bounds_val
    if upper_bound is not None and np.any(params > upper_bound):
        return -_out_of_bounds_val
    # check constraints
    if cons is not None and np.any(cons(params) <= 0):
        return -_out_of_bounds_val

    global _counter
    _counter += 1

    # update builder
    builder = _update_builder(builder, options, params)

    # build graph and compute LD stats
    g = demes.Graph.fromdict(builder)
    sampled_demes = pop_ids

    end_times = {d.name: d.end_time for d in g.demes}
    sample_times = []

    # check for any ancient samples, which have pop id
    # "{deme.name}_sampled_{gen}_{frac_gen}"
    input_sampled_demes = [d for d in sampled_demes]
    for ii, deme in enumerate(sampled_demes):
        if "_sampled_" in deme:
            d = deme.split("_")[0]
            t = float(".".join(deme.split("_")[2:]))
            input_sampled_demes[ii] = d
            sample_times.append(t)
        else:
            sample_times.append(end_times[deme])

    model = compute_bin_stats(g, input_sampled_demes, sample_times=sample_times, rs=rs)

    # normalize statistics based on normalization population
    norm_idx = pop_ids.index(normalization)
    model = moments.LD.Inference.sigmaD2(model, normalization=norm_idx)
    if statistics is None:
        model = moments.LD.Inference.remove_normalized_lds(
            model, normalization=norm_idx
        )
    else:
        model = moments.LD.Inference.remove_nonpresent_statistics(
            model, statistics=statistics
        )

    LL = moments.LD.Inference.ll_over_bins(means, model, varcovs)

    if np.isnan(LL):
        print("God bad result: LL = nan")
        result = _out_of_bounds_val

    # print outputs if verbose > 0
    if verbose > 0 and _counter % verbose == 0:
        param_str = "array([%s])" % (", ".join(["%- 12g" % v for v in params]))
        output_stream.write("%-8i, %-12g, %s%s" % (_counter, LL, param_str, os.linesep))
        moments.Misc.delayed_flush(stream=output_stream, delay=0.5)

    return -LL


def _object_func_log_LD(log_params, *args, **kwargs):
    """
    Objective function for optimization in log(params).
    """
    return _object_func_LD(np.exp(log_params - 1), *args, **kwargs)


def optimize_LD(
    deme_graph,
    inference_options,
    means,
    varcovs,
    pop_ids=None,
    rs=None,
    statistics=None,
    normalization=None,
    maxiter=1000,
    perturb=0,
    verbose=0,
    log=True,
    method="fmin",
    output_stream=sys.stdout,
    output=None,
    overwrite=False,
):
    """
    Optimize demography given a deme graph of initial and fixed parameters and
    inference options that specify which parameters to fit, and bounds or constraints
    on those parameters.

    :param deme_graph: A demographic model as a YAML file in ``demes`` format. This
        should be given as a string specifying the path and file name of the model.
    :param inference_options: A second YAML file, specifying the parameters to be
        optimized, parameter bounds, and constraints between parameters. Please see
        the documentation at
        https://momentsld.github.io/moments/extensions/demes.html#the-options-file
    :param means: The list of average normalized LD and H statistics, as produced by
         the parsing function ``moments.LD.Parsing.bootstrap_data(region_data)``.
    :param varcovs: The list of variance-covariance matrices for data within each
        recombination bin, as produced by the parsing function
        ``moments.LD.Parsing.bootstrap_data(region_data)``.
    :param pop_ids: The list of population names corresponding to the data.
    :param rs: A list of recombination bin edges, defining the recombination
        distance bins.
    :param statistics: A list of two lists, the first being the LD statistics
        present in the data, and the second the list of single-locus statistics
        present in the data.  **WARNING**: This option should *only* be used if
        there are missing or masked statistics in your data, aside from the
        normalizing pi2 statistic!  If there are no missing or removed
        statistics other than the normalization statistic, this option should
        not be used.
    :param normalization: The name of the population that was used to normalize
        the data. See documentation for examples specifying each of these arguments.
    :param maxiter: The maximum number of iterations to run optimization. Defaults
        to 1000. Note: maxiter does not seem to work with the Powell method! This
        appears to be a bug within scipy.optimize.
    :param perturb: The perturbation amount of the initial parameters. Defaults to
        zero, in which case we do not perturb the initial parameters in the input
        demes YAML. If perturb is greater than zero, the initial parameters in the input
        YAML are randomly perturbed by up to `perturb`-fold.
    :param verbose: The frequency to print updates to `output_stream` if greater than 1.
    :param log: If True, optimize over log of the parameters.
    :param method: The optimization method. Available methods are "fmin", "powell",
        and "lbfgsb".
    :output_stream: Defaults to standard output. Can be given an open file stream
        instead or other output stream.
    :param output: If given, the filename for the output best-fit model YAML.
    :param overwrite: If True, overwrites any existing file with the same output
        name.
    :return: List of parameter names, optimized parameter values, and LL
    """
    builder = _get_demes_dict(deme_graph)
    options = _get_params_dict(inference_options)

    if pop_ids is None:
        raise ValueError("Population IDs for sampled demes must be provided")
    if normalization is None:
        raise ValueError("Normalization deme must be provided")
    elif normalization not in pop_ids:
        raise ValueError("Normalizatino deme must be in pop_ids")

    param_names, p0, lower_bound, upper_bound = _set_up_params_and_bounds(
        options, builder
    )
    constraints = _set_up_constraints(options, param_names)

    if not (isinstance(perturb, float) or isinstance(perturb, int)):
        raise ValueError("perturb must be a non-negative number")
    if perturb < 0:
        raise ValueError("perturb must be non-negative")
    elif perturb > 0:
        p0 = _perturb_params_constrained(
            p0, perturb, lower_bound, upper_bound, constraints
        )

    available_methods = ["fmin", "powell", "lbfgsb"]
    if method not in available_methods:
        raise ValueError(
            f"method {method} not available,  must be one of " f"{available_methods}"
        )

    # rescale if log
    if log:
        p0 = np.log(p0) + 1
        obj_fun = _object_func_log_LD
    else:
        obj_fun = _object_func_LD

    args = (
        means,
        varcovs,
        builder,
        options,
        pop_ids,
        rs,
        statistics,
        normalization,
        lower_bound,
        upper_bound,
        constraints,
        verbose,
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
        bounds = list(zip(np.log(lower_bound) + 1, np.log(upper_bound) + 1))
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

    if log:
        xopt = np.exp(xopt - 1)

    if output is not None:
        builder = _update_builder(builder, options, xopt)
        g = demes.Graph.fromdict(builder)
        if overwrite is False and os.path.isfile(output):
            output_stream.write(
                f"Did not write output YAML - {output} already exists. The model is "
                "printed below. To overwrite, set overwrite=True." + os.linesep
            )
            output_stream.write(str(g))
        else:
            demes.dump(g, output)

    return param_names, xopt, fopt


################################################################
# Methods for computing confidence intervals from LD inference #
################################################################

# Cache evaluations of the frequency spectrum inside our hessian/J
# evaluation function
_ld_cache = {}


def _get_godambe_LD(
    func_ex,
    all_boot,
    p0,
    means,
    varcovs,
    eps,
    log=False,
    just_hess=False,
):
    def func(params, means, varcovs):
        key = tuple(params)
        if key not in _ld_cache:
            _ld_cache[key] = func_ex(params)
        y = _ld_cache[key]
        return moments.LD.Inference.ll_over_bins(means, y, varcovs)

    def log_func(logparams, means, varcovs):
        return func(np.exp(logparams), means, varcovs)

    # First calculate the observed hessian
    if not log:
        hess = -moments.LD.Godambe._get_hess(func, p0, eps, args=[means, varcovs])
    else:
        hess = -moments.LD.Godambe._get_hess(
            log_func, np.log(p0), eps, args=[means, varcovs]
        )

    if just_hess:
        return hess

    # Now the expectation of J over the bootstrap data
    J = np.zeros((len(p0), len(p0)))
    # cU is a column vector
    cU = np.zeros((len(p0), 1))
    for bs_ms in all_boot:
        # boot = LDstats(boot)
        if not log:
            grad_temp = moments.LD.Godambe._get_grad(
                func, p0, eps, args=[bs_ms, varcovs]
            )
        else:
            grad_temp = moments.LD.Godambe._get_grad(
                log_func, np.log(p0), eps, args=[bs_ms, varcovs]
            )
        J_temp = np.outer(grad_temp, grad_temp)
        J = J + J_temp
        cU = cU + grad_temp
    J = J / len(all_boot)
    cU = cU / len(all_boot)

    # G = H*J^-1*H
    J_inv = np.linalg.inv(J)
    godambe = np.dot(np.dot(hess, J_inv), hess)
    return godambe, hess, J, cU


def uncerts_LD(
    deme_graph,
    inference_options,
    means,
    varcovs,
    bootstraps=[],
    pop_ids=None,
    rs=None,
    statistics=None,
    normalization=None,
    log=False,
    eps=0.01,
    method="FIM",
    verbose=0,
    output_stream=sys.stdout,
    output=None,
    overwrite=False,
):
    """
    Compute uncertainties for fitted parameters, using the output YAML from
    ``moments.Demes.Inference.optimize_LD()``.
    """
    func_calls = 0

    # Get p0 and parameter information
    builder = _get_demes_dict(deme_graph)
    options = _get_params_dict(inference_options)

    if pop_ids is None:
        raise ValueError("Population IDs for sampled demes must be provided")
    if normalization is None:
        raise ValueError("Normalization deme must be provided")
    elif normalization not in pop_ids:
        raise ValueError("Normalizatino deme must be in pop_ids")

    # when statistics is None, we assume all statistics are present, and
    # we need to remove the normalizing statistic
    if statistics is None:
        statistics = moments.LD.Util.moment_names(len(pop_ids))
        (
            statistics,
            means,
            varcovs,
            bootstraps,
        ) = moments.LD.Godambe._remove_normalized_data(
            statistics, pop_ids.index(normalization), means, varcovs, bootstraps
        )

    param_names, p0, lower_bound, upper_bound = _set_up_params_and_bounds(
        options, builder
    )

    if verbose > 0:
        exp_num_calls = moments.LD.Godambe._expected_number_of_calls(p0)
        output_stream.write(
            "Expected number of function calls: " + str(exp_num_calls) + os.linesep
        )
        moments.Misc.delayed_flush(stream=output_stream, delay=0.5)

        eval_time_1 = -np.inf
        eval_time_2 = np.inf

    def func_ex(params):
        nonlocal builder
        nonlocal options
        nonlocal func_calls
        nonlocal eval_time_1
        nonlocal eval_time_2
        func_calls += 1

        # update the builder dict, then compute LD
        builder = _update_builder(builder, options, params)
        g = demes.Graph.fromdict(builder)

        sampled_demes = pop_ids
        end_times = {d.name: d.end_time for d in g.demes}
        sample_times = []

        # check for any ancient samples, which have pop id
        # "{deme.name}_sampled_{gen}_{frac_gen}"
        input_sampled_demes = [d for d in sampled_demes]
        for ii, deme in enumerate(sampled_demes):
            if "_sampled_" in deme:
                d = deme.split("_")[0]
                t = float(".".join(deme.split("_")[2:]))
                input_sampled_demes[ii] = d
                sample_times.append(t)
            else:
                sample_times.append(end_times[deme])

        model = compute_bin_stats(
            g, input_sampled_demes, sample_times=sample_times, rs=rs
        )

        # normalize statistics based on normalization population
        norm_idx = pop_ids.index(normalization)
        model = moments.LD.Inference.sigmaD2(model, normalization=norm_idx)
        model = moments.LD.Inference.remove_nonpresent_statistics(
            model, statistics=statistics
        )

        if (verbose > 0) and (func_calls % verbose == 0):
            if func_calls == 1:
                eval_time_1 = time.time()
            if func_calls == 2:
                eval_time_2 = time.time()
                eval_time = eval_time_2 - eval_time_1
                output_stream.write(
                    f"One iteration took {eval_time:.1f} seconds, "
                    f"expected total time: {eval_time*exp_num_calls/60:.0f} minutes"
                    + os.linesep
                )
            if func_calls % verbose == 0:
                output_stream.write(
                    f"Finished iteration {func_calls: > {len(str(exp_num_calls))}} "
                    + f"of {exp_num_calls}"
                    + os.linesep
                )
            moments.Misc.delayed_flush(stream=output_stream, delay=0.5)

        return model

    if method == "FIM":
        # computing
        if len(bootstraps) > 0:
            warnings.warn(
                "FIM method chosen but bootstrap replicates provided - "
                "Bootstraps will not be used"
            )
        H = _get_godambe_LD(
            func_ex, [], p0, means, varcovs, eps, log=log, just_hess=True
        )
        uncerts_out = np.sqrt(np.diag(np.linalg.inv(H)))
    elif method == "GIM":
        # check that data and boostraps match in sizes
        if len(bootstraps) == 0:
            raise ValueError(
                "A list of LD bootstrap replicates must be provided to use GIM method"
            )
        else:
            for bs in bootstraps:
                if len(bs) != len(means):
                    raise ValueError(
                        "mismatch in number of bins between bootstrap and data means"
                    )
                for b, d in zip(bs, means):
                    if len(b) != len(d):
                        raise ValueError(
                            "mismatch in number of statistics between boostrap and "
                            "data means"
                        )
        GIM, H, J, cU = _get_godambe_LD(
            func_ex,
            bootstraps,
            p0,
            means,
            varcovs,
            eps,
            log=log,
        )
        uncerts_out = np.sqrt(np.diag(np.linalg.inv(GIM)))
    else:
        raise ValueError("method must be either 'FIM' or 'GIM'")

    if output is not None:
        _write_uncerts_output(
            param_names, p0, uncerts_out, output, overwrite, output_stream
        )

    return uncerts_out
