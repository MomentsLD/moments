============
Installation
============

``moments`` now supports Python 3. Because Python has discontinued support for
Python 2, we do not actively ensure that moments remains fully compatible with
Python 2, and strongly recommend using Python 3.

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

Alternatively, you can clone the git repository

.. code-block:: bash

   git clone https://github.com/MomentsLD/moments.git


and then from within the moments directory (``cd moments``), run

.. code-block:: bash

   pip install -r requirements.txt
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

If you are building ``moments`` from source, you will need to first install
a handful of dependencies. Minimally, these include

- numpy

- scipy

- cython

- mpmath

- demes


All dependencies are listed in `requirements.txt`, and can be installed
together using

.. code-block:: bash

   python -m pip install -r requirements.txt

We also strongly recommend installing ``ipython`` for interactive analyses.

If you are using conda, all dependencies can be installed by navigating to the
moments directory and then running

.. code-block:: bash

   conda install --file requirements.txt

Once dependencies are installed, to install ``moments``, run the following
commands in the moments directory:

.. code-block:: bash

    python -m pip install -e .

Note that you might need sudo privileges to install in this way.

You should then be able to import ``moments`` in your python scripts. Entering
an ipython or python session, type ``import moments``. If, for any reason, you
have trouble installing ``moments`` after following these steps, please submit
an `Issue <https://github.com/MomentsLD/moments/issues>`_.

If you use ``Parsing`` from ``moments.LD``, which reads VCF-formatted files and
computes LD statistics to compare to predictions from ``moments.LD``, you will
need to additionally install

- hdf5

- scikit-allel

- pandas
