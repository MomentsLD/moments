.. _sec_sfs_api:

==============================
API for site frequency spectra
==============================

The Spectrum object
-------------------

.. autoclass:: moments.Spectrum
    :members:

Miscellaneous functions
-----------------------

.. autofunction:: moments.Misc.perturb_params

.. autofunction:: moments.Misc.make_data_dict_vcf

.. autofunction:: moments.Misc.count_data_dict

.. autofunction:: moments.Misc.bootstrap

Demographic functions
---------------------

.. automodule:: moments.Demographics1D
    :members:

.. automodule:: moments.Demographics2D
    :members:

.. automodule:: moments.Demographics3D
    :members:

Inference functions
-------------------

.. autofunction:: moments.Inference.ll

.. autofunction:: moments.Inference.ll_multinom

.. autofunction:: moments.Inference.optimal_sfs_scaling

.. autofunction:: moments.Inference.optimally_scaled_sfs

.. autofunction:: moments.Inference.linear_Poisson_residual

.. autofunction:: moments.Inference.Anscombe_Poisson_residual

.. autofunction:: moments.Inference.optimize_log

.. autofunction:: moments.Inference.optimize_log_fmin

.. autofunction:: moments.Inference.optimize_log_powell

.. autofunction:: moments.Inference.optimize_log_lbfgsb

.. _sec_sfs_api_plotting:

Uncertainty functions
---------------------

.. automodule:: moments.Godambe
    :members: FIM_uncert, GIM_uncert, LRT_adjust

Plotting features
-----------------

Single-population plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: moments.Plotting
    :members: plot_1d_fs, plot_1d_comp_Poisson, plot_1d_comp_multinom
    :noindex:

Multi-population plotting
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: moments.Plotting
    :members: plot_single_2d_sfs, plot_2d_resid, plot_2d_comp_multinom,
        plot_2d_comp_Poisson, plot_3d_comp_multinom, plot_3d_comp_Poisson,
        plot_3d_spectrum, plot_4d_comp_multinom, plot_4d_comp_Poisson
    :noindex:
