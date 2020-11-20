==============================
API for site frequency spectra
==============================

The Spectrum object
-------------------

.. autoclass:: moments.Spectrum
    :members: project, marginalize, swap_axes, log, fold, unfold,
        split, admix, pulse_migrate, integrate,
        Fst, S, Watterson_theta, theta_L, Zengs_E, pi, Tajima_D,
        from_file, to_file, fixed_size_sample, sample,
        from_data_dict, from_demes

.. autofunction:: moments.Misc.make_data_dict_vcf

Demographic functions
---------------------

.. autofunction:: moments.Demographics1D.snm

.. autofunction:: moments.Demographics1D.two_epoch

.. autofunction:: moments.Demographics1D.growth

.. autofunction:: moments.Demographics1D.bottlegrowth

.. autofunction:: moments.Demographics1D.three_epoch

.. autofunction:: moments.Demographics2D.split_mig

.. autofunction:: moments.Demographics2D.IM

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

Plotting features
-----------------
