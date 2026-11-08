Running the tests

The test suite uses pytest. Install moments and the test requirements, then run
the tests from the repository root:

    pip install .
    pip install -r requirements/tests.txt
    python -m pytest tests

To run tests in parallel (using pytest-xdist), as in CI:

    python -m pytest -n auto tests

To run a single test file, or a single test:

    python -m pytest tests/test_Spectrum.py
    python -m pytest tests/test_Spectrum.py -k test_name

Warnings are treated as errors (see `[tool.pytest.ini_options]` in
pyproject.toml), so new deprecation warnings from dependencies will cause tests
to fail.

Tests that need optional dependencies, such as matplotlib, should skip when the
dependency isn't installed, e.g. with `pytest.importorskip("matplotlib")`.
