================================
Specifying models with ``demes``
================================
.. jupyter-kernel:: python3

**New** in version 1.1, ``moments`` can compute the SFS and LD statistics
directly from a ``demes``-formatted demographic model. To learn about how to
describe a demographic model using ``demes``, head to the `demes repository
<https://github.com/popsim-consortium/demes-python>`_ or `documentation
<https://popsim-consortium.github.io/demes-docs/main/index.html>`_ to learn
about specifying multi-population demographic models using ``demes``.

******************
What is ``demes``?
******************

Demographic models specify the historical size changes, migrations, splits and
mergers of related populations. Specifying demographic models using ``moments``
or practically any other simulation engine can become very complicated and
error prone, especially when we want to model more than one population (e.g.
[Ragsdale]_). Even worse, every individual software has its own language and
methods for specifying a demographic model, so a user has to reimplement the
same model across multiple software, which nobody enjoys. To resolve these
issues of reproducibility, replication, and susceptibility to errors, ``demes``
provides a human-readable specification of complex demography that is designed
to make it easier to implement and share models and to be able to use that
demography with multiple simulation engines.

``Demes`` models are written in YAML, and they are then automatically parsed to
create an internal representation of the demography that is readable by
``moments``. ``moments`` can then iterate through the epochs and demographic
events in that model and compute the SFS or LD.

*************************************************
Simulating the SFS and LD using a ``demes`` model
*************************************************

Computing expectations for the SFS or LD using a ``demes`` model is designed to
be as simple as possible. In fact, there is no need for the user to specify any
demographic events or integrate the SFS or LD objects. ``moments`` does all of
that for you.

It's easiest to see the functionality through example. In the tests directory,
there is a YAML description of the [Gutenkunst]_ Out-of-African model:

.. literalinclude:: ../../tests/test_files/gutenkunst_ooa.yml
    :language: yaml

This model describes all the populations (demes), their sizes and times of
existence, their relationships to other demes (ancestors and descendents), and
migration between them. To simulate using this model, we just need to specify
the populations that we want to sample lineages from, the sample size in each
population, and (optionally) the time of sampling. If sampling times are not
given we assume we sample at present time. Ancient samples can be specified by
setting sampling times greater than 0.

Let's simulate 10 samples from each YRI, CEU, and CHB:

.. jupyter-execute::

    import moments
    ooa_model = "../tests/test_files/gutenkunst_ooa.yml"

    sampled_demes = ["YRI", "CEU", "CHB"]
    sample_sizes = [10, 10, 10]

    fs = moments.Spectrum.from_demes(
        ooa_model, sampled_demes=sampled_demes, sample_sizes=sample_sizes
    )

    print(fs.pop_ids)
    print(fs.sample_sizes)
    print("FST(CEU, CHB) =", f"{fs.marginalize([0]).Fst():.3}")

It's that simple. We can also simulate data for a subset of the populations,
while still accounting for migration with other non-sampled populations:

.. jupyter-execute::

    sampled_demes = ["YRI"]
    sample_sizes = [40]

    fs = moments.Spectrum.from_demes(
         ooa_model, sampled_demes=sampled_demes, sample_sizes=sample_sizes
    )
 
    print(fs.pop_ids)
    print(fs.sample_sizes)
    print("Tajima's D =", f"{fs.Tajima_D():.3}")

Or sample a combination of ancient and modern samples from a population:

.. jupyter-execute::

    sampled_demes = ["CEU", "CEU"]
    sample_sizes = [10, 10]
    # sample 10 from present, 10 from 20,000 years ago
    sample_times = [0, 20000]

    fs = moments.Spectrum.from_demes(
         ooa_model,
         sampled_demes=sampled_demes,
         sample_sizes=sample_sizes, 
         sample_times=sample_times
    )

    print(fs.pop_ids)
    print(fs.sample_sizes)
    print("FST(current, ancient) =", f"{fs.Fst():.3}")


We can similarly compute :ref:`LD statistics <sec_ld>`:

.. jupyter-execute::

    import moments.LD
    
    sampled_demes = ["YRI", "CEU", "CHB"]
    y = moments.LD.LDstats.from_demes(
        ooa_model, sampled_demes=sampled_demes, rho=[0, 1, 2]
    )

    print(y.num_pops)
    print(y.pop_ids)

***********************************
Using ``Demes`` to infer demography
***********************************

Above, we showed how to use ``moments`` and ``demes``-based demographic models
to compute expectations for *static* demographic models. That is, given a fixed
demography we can compute expectations for the SFS or LD. We often want to
optimize the parameters of a given demographic model to fit observations from
data. The general idea is that we specify a parameterized model, compute the
expected SFS under that model and it's likelihood given the data, and then
update the model parameters to improve the fit. ``Moments`` uses ``scipy``'s
`optimization functions
<https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ to perform
optimization.

To run the inference, we need three items: 1) the data (SFS) to be fit, 2)
a parameterized demographic model, and 3) a way to tell the optimization
function which parameters to fit along with other options and rules for the
optimizer to follow. We'll assume you already have a data SFS, with the
``pop_ids`` attribute given, saved using ``data.tofile("saved_data.fs")``. For
example, the data could be a 3-dimensional SFS for the three sampled
populations in the Out-of-Africa demography above, so that ``data.pop_ids
= ["YRI", "CEU", "CHB"]``.

The second item is the ``demes``-formatted demographic model, such as the model
written above. In this model, the parameter values are the times, sizes, and
migration rates, and the YAML file specifies all fixed parameters or the
initial guesses for the parameters to be fit.

The third item is a separate YAML-formatted file that tells the optimization
function the variable parameters, any inequality constraints on the parameter
values and other optional arguments to pass to the optimization function.
``moments`` will read this YAML file into a dictionary using a YAML parser, so
it needs to be valid and properly formatted YAML code. The only required field
in the "options" YAML is ``parameters``. For each parameter to be fit, we must
name that parameter, which can be any unique string, and we need to specify
which values in the deme graph it points to. We can optionally include
a description for our own sake, but the optimizer ignores that field. For
example, to fit the bottleneck size in the Out-of-Africa model, we would
include:

.. code-block:: YAML

    parameters:
    - name: N_B
      descrption: Bottleneck size for Eurasian populations
      values:
      - demes:
          OOA:
            epochs:
              0: start_size

This says that the start size of the first (only) epoch of the OOA deme in the
deme graph should be fit. All additional parameters to be fit are included here
under ``parameters``. Any parameter that is not included here is assumed to be
a fixed parameter, and it will remain the value given in the deme graph.

The same parameter can affect multiple values in the deme graph. The size of
the African population in the Out-of-Africa model is applied to both the AMH
and the YRI demes. This simply requires adding additional keys in the
``values`` entry:

.. code-block:: YAML

    parameters:
    - name: N_A
      description: Expansion size
        values:
        - demes:
            AMH:
              epochs:
                0: start_size
            YRI:
              epochs:
                0: start_size

Migration rates can be specified to be fit as well. Note that the index of the
migration is given, pointing to the migrations in the order they are specified
in the demes file.

.. code-block:: YAML

    parameters:
    - name: m_Af_Eu
      description: Symmetric migration rate between Afr and Eur populations
      upper_bound: 1e-3
      values:
      - migrations:
          1: rate

Note here that we have specified the upper bound to be ``1e-3`` (the units are
migrants per generation in this model). For any parameter, we can set the lower
bound and upper bound as shown here. If they are not given, the lower bound
defaults to 0 and the upper bound defaults to infinity.

Finally, we can also specify constraints on parameters. For example, if some
event necessarily occurs before another, we can add that relationship to the
list of constraints.

.. code-block:: YAML

    parameters:
    - name: TA
      description: Time before present of ancestral expansion
      values:
      - demes:
          ancestral:
            epochs:
              0: end_time
    - name: TB
      description: Time of YRI-OOA split
      values:
      - demes:
          AMH:
            epochs:
              0: end_time
    - name: TF
      description: Time of CEU-CHB split
      values:
      - demes:
          OOA:
            epochs:
              0: end_time
    constraints:
    - params: [TA, TB]
      constraint: greater_than
    - params: [TB, TF]
      constraint: greater_than

This specifies each of the event timings in the OOA model to be fit, and the
constraints say that ``TA`` must be greater than ``TB``, and ``TB`` must be
greater than ``TF``.

Additional options can be passed to the optimization function using either
keyword arguments in the ``moments.Demes.Inference.optimize`` function, or as
entries in this options YAML. These include

- ``maxiter``: Maximum number of iterations to run optimization. Defaults to
  1,000.
- ``perturb``: Defaults to 0 (no perturbation of initial parameters). If
  greater than zero, it perturbs the initial parameters by that average
  percentage, with larger values resulting in stronger perturbation of initial
  guesses.
- ``verbose``: Defaults to 0. If greater than zero, it prints an update to the
  specified ``output_stream`` (which defaults to ``sys.stdout``) every
  ``verbose`` iterations.
- ``uL``: Defaults to None. If given, this is the product of the per-base
  mutation rate and the length of the callable genome used to compile the data
  SFS. If we don't give this scaled mutation rate, we optimize with theta as
  a free parameter. Otherwise, we optimize with theta given by :math:`\theta=4
  \times N_e \times uL`, and :math:`N_e` is taken to be the size of the
  root/ancestral deme (for which the size can be a either be a fixed parameter
  or a parameter to be fit!).
- ``log``: Defaults to True. If True, optimize the log of the parameters.
- ``method``: The optimization method to use, currently with the options "fmin"
  (Nelder-Mead), "powell", or "lbfgsb". Defaults to "fmin".


Single-population inference example
===================================

To demonstrate, we'll fit a simple single-population demographic model to the
synonymous variant SFS in the Mende (MSL) from the Thousand Genomes data. The
data for this population is stored in the docs/data directory. We previous
parsed all coding variation and used a mutation model to estimate
:math:`u\times L`.

.. jupyter-execute::

    import moments
    import pickle

    all_data = pickle.load(open("./data/msl_data.bp", "rb"))
    data = all_data["spectra"]["syn"]
    data.pop_ids = ["MSL"]
    # infer using the folded SFS, since we aren't accounting for mispolarization here
    data = data.fold()
    uL = all_data["rates"]["syn"]
    print("scaled mutation rate:", uL)

We'll fit a demographic model that includes an ancient expansion and a more
recent exponential growth. This initial model is stored in the docs/data
directory as well.

The YAML specification of this model is

.. literalinclude:: ../data/msl_initial_model.yml
   :language: YAML

And we can specify that we want to fit the times of the size changes, and all
population sizes. (Note that if we did not have an estimate for the mutation
rate, we would not fit the ancestral size.)

.. literalinclude:: ../data/msl_options.yml
   :language: YAML

.. jupyter-execute::

    deme_graph = "./data/msl_initial_model.yml"
    options = "./data/msl_options.yml"

And now we can run the inference:

.. jupyter-execute::

    output = "./data/msl_best_fit_model.yml"
    ret = moments.Demes.Inference.optimize(
        deme_graph, options, data, uL=uL, output=output, overwrite=True)
    param_names, opt_params, LL = ret
    print("Log-likelihood:", -LL)
    print("Best fit parameters")
    for n, p in zip(param_names, opt_params):
        print(f"{n}\t{p:.3}")

.. note:: We can tell the optimization function to write a new deme graph YAML
    by passing the option ``output="new_deme_graph.yml"``, and can specify whether
    to overwrite an existing file with that file path and name by setting
    ``overwrite=True``.

.. todo:: Plot the inferred demographic model, using some upcoming demes plotting
    tools that Graham Gower is developing.

We can see how well our best fit model fits the data, using ``moments``'
plotting features.

.. jupyter-execute::

    import matplotlib.pylab as plt
    fs = moments.Spectrum.from_demes(output, ["MSL"], data.sample_sizes)
    moments.Plotting.plot_1d_comp_multinom(fs, data, show=False)

**********
References
**********

.. [Gutenkunst]
    Gutenkunst, Ryan N., et al. "Inferring the joint demographic history of
    multiple populations from multidimensional SNP frequency data."
    *PLoS genet* 5.10 (2009): e1000695.

.. [Ragsdale]
    Ragsdale, Aaron P., et al. "Lessons learned from bugs in models of human
    history." *The American Journal of Human Genetics* 107.4 (2020): 583-588.
