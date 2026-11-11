# Contributing to TimeSHAP

## Set up

TimeSHAP supports Python 3.10 to 3.14. From a clone of the repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Run the tests

```bash
python -m pytest
```

The tests use mocked models and run in a few seconds. CI also runs them on Windows, on macOS, and with the oldest
supported dependency versions (`uv pip install --resolution lowest-direct ".[test]"`).

To run a notebook the way CI does:

```bash
pip install -e ".[notebooks]"
cd notebooks/AReM
python -m jupyter nbconvert --to notebook --execute --output-dir /tmp/executed AReM.ipynb
```

`AReM_TF.ipynb` also needs `pip install tensorflow`.

## Pull requests

- Add a test that fails without your change. A bug fix includes a test that reproduces the bug.
- Keep explanations reproducible. If a change alters the Shapley values TimeSHAP returns for the same inputs and seed,
  say so in the pull request and in `CHANGELOG.md`.
- Add an entry to `CHANGELOG.md` for any change a user can notice.

## Releases

1. Update `CHANGELOG.md` and the version in `src/timeshap/version.py`.
2. Merge to `main`. The `Publish to PyPI` workflow runs the tests, builds the package and publishes it to PyPI with the
   `PYPI_API_TOKEN` secret.
