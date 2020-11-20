=============
SFS Inference
=============

*********************
Computing likelihoods
*********************

Following `Sawyer and Hartl (1992) <https://www.genetics.org/content/132/4/1161.short>`_
the distribution of mutation frequencies is treated as a Poisson random field, so
that composite likelihoods (in which we assume mutations are independent) are computed
by taking Poisson likelihoods over bins in the SFS. We typically work with
log-likelihoods, so that the log-likelihood of the data (:math:`D`) given the model
(:math:`M`) is

.. math::
    \log{\mathcal{L}} = \sum_{i} D_i \log{M_i} - M_i - \log{D_i !}

where :math:`i` indexes the bins of the SFS.

Likelihoods can be computed from ``moments.Inference``:

.. jupyter-execute::

    import moments

    theta = 1000
    model = theta * moments.Demographics1D.snm([10])

    data = model.sample()

    print(model)
    print(data)

.. jupyter-execute::

    print(moments.Inference.ll(model, data))

When simulating under some demographic model, we usually use the default ``theta``
of 1, because the SFS scales linearly in the mutation rate. When comparing to data
in this case, we need to rescale the model SFS. It turns out that the
maximum-likelihood rescaling is that which makes the total number of segregating
sites in the model equal to the total number in the data:

.. jupyter-execute::

    data = moments.Spectrum([0, 3900, 1500, 1200, 750, 720, 600, 400, 0])
    model = moments.Demographics1D.two_epoch((2.0, 0.1), [8])

    print("Number of segregating sites in data:", data.S())
    print("Number of segregating sites in model:", model.S())
    print("Ratio of segregating sites:", data.S() / model.S())

    opt_theta = moments.Inference.optimal_sfs_scaling(model, data)
    print("Optimal theta:", opt_theta)

Then we can compute the log-likelihood of the rescaled model with the data, which
will give us the same answer as ``moments.Inference.ll_multinom`` using the unscaled
data:

.. jupyter-execute::

    print(moments.Inference.ll(opt_theta * model, data))
    print(moments.Inference.ll_multinom(model, data))

************
Optimization
************

``moments`` optimization is effectively a wrapper for ``scipy`` optimization
routines, with some features specific to working with SFS data. In short, given
a demographic model defined by a set of parameters, we try to find those parameters
that minimize the negative log-likelihood of the data given the model. There are
a number of optimization functions available in ``moments.Inference``:

- ``optimize`` and ``optimize_log``:
- ``optimize_lbfgsb`` and ``optimize_log_lbfgsb``:
- ``optimize_log_fmin``:
- ``optimize_powell`` and ``optimize_log_powell``:

With each method, we require at least three inputs: 1) the initial guess, 2) the
data SFS, and 3) the model function that returns a SFS of the same size as the data.

Additionally, it is common to set the following:

- ``lower_bound`` and ``upper_bound``:
- ``fixed_params``:
- ``verbose``:

For a full description of the various inference functions, please see the *SFS inference
API*.

Example
-------

Here, we've created some fake data for a two-population split-migration model, and
we reinfer the input parameters to the model used to create that data. This example
uses the ``optimize_log_fmin`` optimization function.

.. jupyter-execute::

    input_theta = 10000
    params = [2.0, 3.0, 0.2, 2.0]
    model_func = moments.Demographics2D.split_mig
    model = model_func(params, [20, 20])
    model = input_theta * model
    data = model.sample()

    p_guess = [2, 2, .1, 4]
    lower_bound = [1e-3, 1e-3, 1e-3, 1e-3]
    upper_bound = [10, 10, 1, 10]

    p_guess = moments.Misc.perturb_params(
        p_guess, lower_bound=lower_bound, upper_bound=upper_bound)

    opt_params = moments.Inference.optimize_log_fmin(
        p_guess, data, model_func,
        lower_bound=lower_bound, upper_bound=upper_bound,
        verbose=20) # report every 20 iterations

    print("Input parameters:", params)
    print("Refit parameters:", opt_params)

    print("Input theta:", input_theta)
    print("Refit theta:",
        moments.Inference.optimal_sfs_scaling(
            model_func(opt_params, data.sample_sizes),
            data))

    moments.Plotting.plot_2d_comp_multinom(
        model_func(opt_params, data.sample_sizes), data)
