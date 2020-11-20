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

Plotting features
-----------------
