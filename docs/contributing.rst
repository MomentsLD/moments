.. _sec_contributing:

============
Contributing
============

Contributions to ``moments`` are welcome, whether they are bug fixes, new
features, improvements to the documentation, or new tests. If you are planning
a large change, please first open an `Issue
<https://github.com/MomentsLD/moments/issues>`_ to discuss it, so that we can
agree on an approach before you put in the work.

Overview
========

1. Fork the `moments repository <https://github.com/MomentsLD/moments>`_ on
   GitHub.
2. Create a branch off of ``devel`` for your changes.
3. Make your changes, with tests and documentation, formatted with ``black``.
4. Add a brief entry to the change log.
5. Open a pull request (PR) against the ``devel`` branch of
   ``MomentsLD/moments``.

PRs should target ``devel``, not ``main``. The ``main`` branch is only updated
when a new version is released.

Setting up a development environment
====================================

Fork the repository on GitHub, then clone your fork and add the main
repository as a remote named ``upstream``:

.. code-block:: bash

    git clone git@github.com:<your-username>/moments.git
    cd moments
    git remote add upstream https://github.com/MomentsLD/moments.git

Install the dependencies, ``moments`` itself (as an editable install), and the
requirements for testing and formatting:

.. code-block:: bash

    python -m pip install -r requirements.txt
    python -m pip install -e .
    python -m pip install -r requirements/tests.txt -r requirements/lint.txt

``moments`` includes compiled Cython extensions, so after changing any ``.pyx``
files you need to run ``python -m pip install -e .`` again to rebuild them. See
:ref:`Installation <sec_installation>` for more details on building from
source.

Making changes
==============

Start each set of changes from an up-to-date copy of ``devel``, on a new
branch:

.. code-block:: bash

    git fetch upstream
    git checkout -b my-branch-name upstream/devel

Please keep each PR focused on a single bug fix or feature. Separate changes
are easier to review in separate PRs. If your PR addresses an open issue,
reference it in the PR description (e.g., "Fixes #123").

If ``devel`` changes while you are working, update your branch with
``git fetch upstream`` followed by ``git rebase upstream/devel`` (or
``git merge upstream/devel``).

Code formatting
---------------

All Python code must be formatted with `black
<https://black.readthedocs.io/>`_. The black version is pinned in
``requirements/lint.txt`` and ``pyproject.toml``, because black's style changes
from year to year. Before committing, format the files that you changed:

.. code-block:: bash

    black path/to/changed_file.py

or check the whole repository with ``black --check .`` from the repository
root.

Tests
-----

Bug fixes should include a test that fails without the fix, and new features
should include tests of their behavior. Tests live in the ``tests`` directory
and are run with ``pytest``:

.. code-block:: bash

    python -m pytest -n auto tests

A few things to be aware of:

- Warnings are treated as errors (see ``[tool.pytest.ini_options]`` in
  ``pyproject.toml``), so code that raises a warning, including a deprecation
  warning from a dependency, will fail the tests.
- Tests that need an optional dependency, such as ``matplotlib``, should be
  skipped when it isn't installed, for example with
  ``pytest.importorskip("matplotlib")``.
- Keep tests fast. They are run many times in CI.

Documentation
-------------

New functions and arguments need docstrings, which are used to build the API
documentation. Changes that users will notice should also be reflected in the
relevant pages under ``docs/``. Code examples in the documentation are executed
when the docs are built, so they must run without errors. To build the docs
locally:

.. code-block:: bash

    python -m pip install sphinx sphinx_rtd_theme jupyter-sphinx
    cd docs
    sphinx-build . _build

Change log
----------

Add an entry describing your change to the :ref:`change log <sec_changelog>`
in ``docs/introduction.rst``, under the heading for the upcoming version at
the top of the list. Entries are brief, a single line for each change, and
describe what changed from a user's point of view. Credit yourself with your
GitHub username, for example:

.. code-block:: rst

    - Fix bug in ``moments.Spectrum.project`` for masked spectra (@your-username)

Minor internal changes, such as fixing a deprecation warning, do not need an
entry.

Using AI tools
--------------

You are welcome to use AI coding assistants when contributing, and they can be
very helpful for tracking down bugs, writing tests, and checking documentation.
However, you are responsible for every line in your PR, and we expect a human
to have carefully reviewed all of it before it is submitted. Before opening a
PR, make sure that the code:

1. **Is readable.** You understand what it does and why, and can explain it
   during review. Another person should be able to follow it.
2. **Is documented.** New functions and arguments have accurate docstrings,
   and non-obvious choices are explained in comments.
3. **Doesn't clutter the code.** Changes are limited to what the PR is meant
   to do, without unrelated edits, unnecessary abstractions, or redundant
   comments.
4. **Passes a human sanity check.** The results make sense to you. AI tools
   can be confidently wrong, so check their claims (for example, about the
   cause of a bug or how a dependency behaves) instead of taking them on
   faith.

Please mention in the PR description if you used AI tools substantially, so
that reviewers know to look closely. We will ask for changes, or close the PR,
if it looks like it hasn't been reviewed by its author.

Opening a pull request
======================

Push your branch to your fork and open a PR on GitHub, with ``devel`` of
``MomentsLD/moments`` as the base branch:

.. code-block:: bash

    git push origin my-branch-name

In the PR description, explain what the change does and why, and reference any
related issues.

Opening the PR runs our continuous integration checks:

- **Formatting**: ``black`` runs first, and all other checks wait for it to
  pass, so a formatting error fails quickly.
- **Tests**: the test suite runs on Linux and macOS, using the two most recent
  Python versions that ``moments`` supports.
- **Documentation**: the docs are built, which runs all code examples in them.
- **Wheels**: wheels are built and tested for all supported Python versions
  (see ``requires-python`` in ``pyproject.toml``).

All checks need to pass before a PR is merged. If a check fails, click through
to its log on GitHub to see the error, fix it on your branch, and push again.
The PR updates automatically.
