============================
Two-locus frequency spectrum
============================

See :ref:`two-locus-usage` for introduction and examples to the Two-Locus extension.

Parameters and selection models
===============================

``moments.TwoLocus`` models a single population. Times are measured in units of
:math:`2N_e` generations and sizes are relative to the ancestral :math:`N_e`. The
recombination rate between the two loci is ``rho`` :math:`= 4N_e r`, and the mutation
rate at each locus under the default infinite sites model is
``theta`` :math:`= 4N_e u`.

Selection is specified in one of three ways, and at most one may be given:

- ``gamma``: additive selection with the same coefficient
  :math:`\gamma = 2N_e s` at both loci and no epistasis. This is shorthand for
  ``sel_params = [2 * gamma, gamma, gamma]``, so that the single-locus spectrum at each
  locus matches the single-site SFS with selection coefficient :math:`\gamma` and
  :math:`h = 1/2`. (With the reversible mutation model, ``finite_genome=True``,
  ``gamma`` is instead additive selection at the left locus only.)
- ``sel_params = [gamma_AB, gamma_A, gamma_B]``: additive selection within loci, with
  scaled selection coefficients for the `AB`, `Ab`, and `aB` haplotypes relative to
  `ab`. For example, selection at only the left locus is ``[gamma, gamma, 0]``, and
  there is epistasis if ``gamma_AB != gamma_A + gamma_B``.
- ``sel_params_general``: scaled selection coefficients for each of the nine diploid
  two-locus genotypes relative to `ab/ab`, which allows for dominance.

The helper functions in ``moments.TwoLocus.Util`` (below) construct ``sel_params`` and
``sel_params_general`` for common selection models.

API
===

The ``TLSpectrum`` class handles all manipulations of the two-locus frequency spectrum:

.. autoclass:: moments.TwoLocus.TLSpectrum
    :members:

The ``Demographics`` module contains some standard demographic models. This is a good
place to look for some inspiration to create your own two-locus models as well.

.. automodule:: moments.TwoLocus.Demographics
    :members:

The ``Util`` module has helper functions for selection models and for computing
low-order statistics from the two-locus frequency spectrum:

.. automodule:: moments.TwoLocus.Util
    :members:

The ``Inference`` module has functions for computing likelihoods and fitting
two-locus models to data:

.. automodule:: moments.TwoLocus.Inference
    :members: ll, ll_multinom, optimal_sfs_scaling, optimize_log_fmin
