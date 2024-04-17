============
Installation
============

``moments`` now supports Python 3. Because Python is soon discontinuing support for
Python 2, we do not actively ensure that moments remains fully compatable with Python
2, and strongly recommend using Python 3.

Using pip
=========

.. todo::
    Update docs when moments is installable via pip, as ``pip install moments-popgen``.

A simple way to install ``moments`` is via ``pip``. ``numpy``, ``mpmath``, and ``cython``
are install requirements, but installing ``moments`` directly from the git repository
using ``pip`` should install these dependencies automatically:

.. code-block:: bash

   pip install git+https://github.com/MomentsLD/moments.git

This approach can also be used to install the development branch of ``moments``:

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

``moments`` is available via `Bioconda <https://anaconda.org/bioconda/moments>`_.

The most recent release of ``moments`` can be installed by running

.. code-block:: bash

   conda install -c bioconda moments

The `conda channels <https://bioconda.github.io/user/install.html#set-up-channels>`_
must be set up to include bioconda, which can be done by running

.. code-block:: bash
   
   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge



Dependencies and details
========================

``moments`` and ``moments.LD`` requires a number of dependencies. Minimally,
these include

- numpy

- scipy

- cython

- mpmath

- demes


All dependencies are listed in `requirements.txt`, and can be install together
using

.. code-block:: bash

   python -m pip install -r requirements.txt

We also strongly recommend installing ``ipython`` for interactive analyses.

If you are using conda, all dependencies can be installed by navigating to the
moments directory and then running

.. code-block:: bash

   conda install --file requirements.txt

Once dependencies are installed, to install ``moments``, run the following commands
in the moments directory:

.. code-block:: bash

    python -m pip install -e .

Note that you might need sudo privileges to install in this way.

You should then be able to import ``moments`` in your python scripts. Entering an
ipython or python session, type ``import moments``. If, for any reason, you have
trouble installing ``moments`` after following these steps, please submit an
`Issue <https://github.com/MomentsLD/moments/issues>`_.

If you use ``Parsing`` from ``moments.LD``, which reads VCF-formatted files and
computes LD statistics to compare to predictions from ``moments.LD``, you will need to
additionally install

- hdf5

- scikit-allel
