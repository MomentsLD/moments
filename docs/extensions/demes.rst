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

***************************
Briefly, what is ``demes``?
***************************

Demographic models specify the historical size changes, migrations, splits and
mergers of related populations. Specifying demographic models using ``moments``
or practically any other simulation can become very complicated and error
prone, especially when we want to model more than one population (e.g.
[Ragsdale]_). ``demes`` provides a human-readable specification of complex
demography that is designed to make it easier to implement and share models and
to be able to use that demography with multiple simulation engines.

``Demes`` models are written in YAML, and they are then automatically parsed to
create an internal representation of the demography that is readable by
``moments``. ``moments`` can then iterate through the epochs and demographic
events in that model and compute the SFS or LD.

*************************************************
Simulating the SFS and LD using a ``demes`` model
*************************************************

Computing expectations for the SFS or LD using a ``demes`` model is designed to
be as simple as possible. In fact, there is no need for the user to specify any
demographic models or integrate the SFS or LD objects. ``moments`` does all of
that for you.

It's easiest to see the functionality through example. In the tests directory,
there is a YAML description of the Gutenkunst Out-of-African model:

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

It's that simple. We can also simulate data for a subset of the populations, while
still accounting for migration with other non-sampled populations:

.. jupyter-execute::

    sampled_demes = ["YRI"]
    sample_sizes = [40]

    fs = moments.Spectrum.from_demes(
         ooa_model, sampled_demes=sampled_demes, sample_sizes=sample_sizes
    )
 
    print(fs.pop_ids)
    print(fs.sample_sizes)

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


We can similarly compute :ref:`LD statistics <sec_ld>`:

.. jupyter-execute::

    import moments.LD
    
    sampled_demes = ["YRI", "CEU", "CHB"]
    y = moments.LD.LDstats.from_demes(
        ooa_model, sampled_demes=sampled_demes, rho=[0, 1, 2]
    )

    print(y.num_pops)
    print(y.pop_ids)

*******************
Future developments
*******************

Currently, ``moments`` can only use ``demes``-based demographic models to
compute expectations for *static* demographic models. That is, given a fixed
demography we can compute expectations for the SFS or LD. Future versions of
``moments`` and ``demes`` will allow demographic inference to be performed
using a YAML specification, so stay tuned!

**********
References
**********

.. [Ragsdale]
    Ragsdale, Aaron P., et al. "Lessons learned from bugs in models of human
    history." *The American Journal of Human Genetics* 107.4 (2020): 583-588.
