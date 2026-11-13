.. _sec_api_ld:

==============================
API for linkage disequilibrium
==============================

********************************
LD statistics class and function
********************************

.. autoclass:: moments.LD.LDstats
    :members:

.. _sec_api_ld_demog:

*********************
Demographic functions
*********************

.. automodule:: moments.LD.Demographics1D
    :members:

.. automodule:: moments.LD.Demographics2D
    :members:

.. automodule:: moments.LD.Demographics3D
    :members:

*****************
Utility functions
*****************

.. automodule:: moments.LD.Util
    :members:

*****************
Parsing functions
*****************

.. automodule:: moments.LD.Parsing
    :members:

********************************************
Inference and computing confidence intervals
********************************************

Inference methods
_________________

.. automodule:: moments.LD.Inference
    :members:

Confidence intervals
____________________

.. automodule:: moments.LD.Godambe
    :members:

********
Plotting
********

These plotting functions are meant as examples, which can be used as a starting point
for your own ``matplotlib`` plots.

.. automodule:: moments.LD.Plotting
    :members: plot_ld_curves, plot_ld_curves_comp
