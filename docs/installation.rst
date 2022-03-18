============
Installation
============

``moments`` now supports Python 3. Because Python is soon discontinuing support for
Python 2, we do not actively ensure that moments remains fully compatable with Python
2, and strongly recommend using Python 3.

Using pip
=========

The simplest way to install ``moments`` is via ``pip``. Note that ``numpy`` and ``cython``
are install requirements, but installing ``moments`` directly from the git repository
using ``pip`` should install these dependencies automatically:

.. code-block:: bash

   pip install git+https://bitbucket.org/simongravel/moments.git

Alternatively, you can clone the git repository

.. code-block:: bash

   git clone https://bitbucket.org/simongravel/moments.git


and then from within the moments directory (``cd moments``), run

.. code-block:: bash

   pip install numpy, cython
   pip install .


Using bioconda
==============

``moments`` is now available on Bioconda! If you use conda, you can install the most
recent release of ``moments`` by running

.. code-block:: bash

   conda install -c bioconda moments

Dependencies and details
========================

``moments`` and ``moments.LD`` requires a number of dependencies. These are

- numpy

- scipy

- cython

- mpmath

- matplotlib

- networkx

- pandas

Dependencies can be installed using pip. For example to install ``cython``,
run ``pip install cython``. All the dependencies can be installed together using

.. code-block:: bash

   pip install -r requirements.txt

Depending on the python distribution you use, it may be useful to add the directory
to ``cython`` in your python path.

We also strongly recommend installing ``ipython`` for interactive analyses.

If you are using conda, all dependencies can be installed by navigating to the
moments directory and then running

.. code-block:: bash

   conda install --file requirements.txt

Once dependencies are installed, to install ``moments``, run the following commands
in the moments directory:

.. code-block:: bash

   python setup.py build_ext --inplace
   python setup.py install

Note that you might need sudo privileges to install in this way.

You should then be able to import ``moments`` in your python scripts. Entering an
ipython or python session, type ``import moments``. If, for any reason, you have
trouble installing ``moments`` after following these steps, please submit an
`Issue <https://bitbucket.org/simongravel/moments/issues>`_.

If you use ``Parsing`` from ``moments.LD``, which reads VCF-formatted files and
computes LD statistics to compare to predictions from ``moments.LD``, you will need to
additionally install

- hdf5

- scikit-allel

