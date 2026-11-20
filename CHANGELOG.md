# Changelog

## 1.0.5

TimeSHAP installs and runs again with current versions of shap, numpy, pandas,
scipy and IPython. Explanations do not change: for the same inputs, seeds and
shap version, 1.0.5 returns the same values as 1.0.4.

### Fixed

- `timeshap.explainer` imports with shap 0.43 and later, which renamed `Kernel` to `KernelExplainer` (#48, #56).
- Importing `timeshap.plot` before `timeshap.explainer` no longer fails with a circular import (#49).
- `plot_global_feat` works with pandas 2 and later (#62, #63, thanks @daniel-n-silva).
- `calc_avg_event` and `calc_avg_sequence` work with scipy 1.11 and later, and with text outside the feature columns, such
  as an entity column. `calc_avg_event` also accepts text in categorical features.
- `get_avg_score_with_avg_event` works with numpy 2.5 and later.
- The global functions (`prune_all`, `event_explain_all`, `feat_explain_all`, `calc_global_explanations`, `global_report`)
  work without `entity_col`. Each sequence is then identified by its position in the data. Pass `entity_col` when results
  from several runs must match: the random sample `max_instances` takes changes the positions, and so does appending to
  files with `append_to_files` for different data.
- `event_explain_all` and `feat_explain_all` with a tolerance and no `pruning_data` now compute the pruning. They failed
  before.
- A time column given as an index now sorts the sequences. It was ignored when the entity column was given as a name.
- Explanation CSV files are written as UTF-8, so they can be written on Windows (#50).
- The global functions no longer change the option dicts you pass them. Reusing the same dict for a local report after a
  global report failed before.
- The notebooks run with IPython 9 and numpy 2.

### Changed

- Requires Python 3.10 or later. Tested on 3.10 to 3.14.
- Supported dependency ranges: numpy 1.22 to 2.x, pandas 1.4 to 3.x, scipy 1.8 to 1.x, shap 0.42.1 to 0.53, altair 4.2
  to 6.x. shap has an upper bound because TimeSHAP uses its private modules.
- `import timeshap` no longer enables the Feedzai Altair theme for the whole session. To get the look of the README
  images, see [Plots](README.md#plots).
- seaborn, plotly, matplotlib, scikit-learn and feedzai-altair-theme are no longer installed with TimeSHAP. TimeSHAP did
  not use the first four, and scikit-learn still comes with shap.
- PyTorch can be installed with `pip install "timeshap[torch]"`.

### Development

- Packaging moved from `setup.py` to `pyproject.toml`.
- GitHub Actions run the tests on every pull request: Python 3.10 to 3.14 on Linux, 3.12 on Windows and macOS, and the
  oldest supported dependencies. They also run the three notebooks and check the built distribution.
- Releases publish to PyPI only after the tests pass.
- Two tests compared against reference values that no shap version in the supported range produces. They now hold the
  values the code returns on shap 0.42.1 to 0.53.

## Earlier releases

See the [commit history](https://github.com/feedzai/timeshap/commits/main).
