.. _sec_installation:

============
Installation
============

``moments`` requires Python 3.11 or newer (up to 3.15).

Using pip
=========

The simplest way to install the latest stable version of ``moments`` is to

.. code-block:: bash
    
    pip install moments-popgen

This installs ``moments`` along with the minimal dependencies, and you should
be able to ``import moments``.

To install the development version of ``moments``, you can install directly
from the development branch at Github:

.. code-block:: bash

   pip install git+https://github.com/MomentsLD/moments.git@devel

Alternatively, you can clone the git repository and install from source (use
``pip install -e .`` for an editable, development install):

.. code-block:: bash

   git clone https://github.com/MomentsLD/moments.git
   cd moments
   pip install .

Using conda
===========

``moments`` is available via `Bioconda
<https://anaconda.org/bioconda/moments>`_, and can be installed by running

.. code-block:: bash

   conda install -c bioconda moments

The `conda channels
<https://bioconda.github.io/user/install.html#set-up-channels>`_ must be set up
to include bioconda, which can be done by running

.. code-block:: bash
   
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge



Dependencies and details
========================

Installing with ``pip`` installs the required dependencies automatically:

- ``numpy`` (>= 2.0, < 3.0)
- ``scipy`` (>= 1.13)
- ``cython`` (>= 3.1)
- ``mpmath`` (>= 1.0, < 1.4)
- ``demes`` (>= 0.2)

Some features need additional packages:

- Plotting: ``matplotlib``
- Parsing VCFs with ``moments.LD.Parsing``, which computes LD statistics to
  compare to predictions from ``moments.LD``: ``scikit-allel``, ``h5py``, and
  ``pandas``

We also strongly recommend installing ``ipython`` for interactive analyses.

For development, ``requirements.txt`` lists the dependencies along with the
packages needed to build the documentation. From the moments directory, these
can be installed with

.. code-block:: bash

   python -m pip install -r requirements.txt

or, using conda,

.. code-block:: bash

   conda install -c conda-forge --file requirements.txt

You should then be able to import ``moments`` in your python scripts. Entering
an ipython or python session, type ``import moments``. If, for any reason, you
have trouble installing ``moments`` after following these steps, please submit
an `Issue <https://github.com/MomentsLD/moments/issues>`_.
