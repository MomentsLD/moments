# Contributing to moments

Contributions are welcome! The full guide is in the documentation:
[Contributing](https://momentsld.github.io/moments/contributing.html). In
brief:

- **Fork and branch.** Fork this repository and create a branch off of
  `devel` for your changes.
- **Open PRs against `devel`**, not `main`. `main` is only updated for
  releases. Keep each PR to a single bug fix or feature, and reference any
  issue it addresses (e.g., "Fixes #123"). For large changes, please open an
  issue to discuss them first.
- **Format with black.** CI checks formatting with black (version pinned in
  `requirements/lint.txt`) before running anything else. Run `black` on the
  files you changed before committing.
- **Add tests.** Bug fixes should include a test that fails without the fix,
  and new features should be tested. Run the tests with
  `python -m pytest -n auto tests`. Warnings are treated as errors.
- **Update the docs** for changes users will notice. Code examples in the docs
  are executed when the docs are built, so they must run.
- **Add a change log entry** to `docs/introduction.rst`, under the upcoming
  version: a brief, one-line description of the change, crediting yourself,
  e.g. `- Fix bug in moments.Spectrum.project for masked spectra (@your-username)`.
- **AI tools are welcome, but you are responsible for the result.** Review
  every line before submitting: the code should be readable, documented, free
  of unrelated changes or clutter, and make sense to you. Check the AI's claims
  rather than accepting them blindly, and mention substantial AI use in the PR
  description.

To set up a development environment:

```
git clone git@github.com:<your-username>/moments.git
cd moments
git remote add upstream https://github.com/MomentsLD/moments.git
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -r requirements/tests.txt -r requirements/lint.txt
```
