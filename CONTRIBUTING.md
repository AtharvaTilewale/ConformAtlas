# Contributing to ConformAtlas

Thank you for considering a contribution. ConformAtlas is a scientific tool, and
the one rule that overrides all others is that **every numerical component must
be checked against an analytical result or an independent implementation** before
it is described as working. This document explains how to set up a development
environment, run the checks, and add a validation check for new functionality.

## Development installation

```bash
git clone https://github.com/AtharvaTilewale/ConformAtlas.git
cd ConformAtlas
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"          # package + MDAnalysis, pytest, ruff, Sphinx, and the extras below
pip install prody                      # optional: ProDy cross-check of the DCCM in the validation suite
```

MDAnalysis is a core dependency and is installed with the package. The `dev`
extra contains pytest, pytest-cov, ruff, build and twine, the optional-feature
backends `plotly` (interactive figures), `h5py` (HDF5 container) and `rich`
(progress bars) — without them the corresponding tests skip silently — and the
reference implementations `deeptime` (MSM estimators, TICA) and `pymbar`
(statistical inefficiency, MBAR) that the validation checks compare against.
The user-facing extras are `pdf` (WeasyPrint), `interactive` (plotly), `hdf5`
(h5py), `progress` (rich) and `all`. A `plumed` executable on `PATH` enables
the `sum_hills` cross-check of the metadynamics bias; the check is skipped
without it.

GROMACS (`gmx` on `PATH`, 2021 or newer) is needed for the GROMACS backend tests and
for the `gmx covar`/`anaeig`/`sham`/`gyrate` validation rows; those tests and rows are
skipped when it is absent.

## Coding conventions

- Python ≥ 3.10, type hints on public functions, NumPy-style docstrings.
- Units: Å, ps, kJ/mol; `R = 0.008314462618 kJ/(mol K)` (`conformatlas.utils.R_GAS_CONSTANT`).
  The GROMACS backend converts nm → Å at the parsing boundary.
- State every assumption a function makes in its docstring (equal-weight replicates,
  lag-time Markovianity, reversibility, fixed basin labels in the bootstrap, …).
- No placeholders, mocks or TODO stubs in shipped code. If something cannot be
  implemented to the validation standard, leave it out and say so in the pull request.
- Anything that could not be validated against an independent reference must be
  labelled **UNVERIFIED** in its docstring and in `docs/methods.md`, never described as
  validated.

### Lint and format

```bash
ruff check src tests validation
ruff format src tests validation
```

Both must pass; CI runs `ruff check` and `ruff format --check`.

### Tests

```bash
pytest -q                               # unit + integration (integration tests need MDAnalysis)
pytest -q -m "not slow"                 # skip the slow ones
pytest --cov=conformatlas --cov-report=term-missing
```

Unit tests live in `tests/unit/test_<module>.py`, integration tests in
`tests/integration/`. Tests on real data use the 190 kB neuraminidase fixture
`tests/data/h1n1_ca/` (Cα + ligand heavy atoms, 81 frames); do not add large binary
files to the repository.

### Documentation

```bash
python -m sphinx -W -b html docs docs/_build/html
```

`-W` turns warnings into errors and is what CI runs. Pages are Markdown (MyST) under
`docs/`; add new pages to the toctree in `docs/index.rst`. Numbers quoted in
`docs/tutorial_neuraminidase.md` are re-derived from the tutorial's output files by
`validation/checks_docs.py`; if you change a default that changes those numbers,
re-run the tutorial and update the text.

## The scientific validation suite

```bash
python validation/validate.py           # writes validation/VALIDATION.md and results.json
python validation/checks_docs.py        # documentation checks
```

`validate.py` runs the core checks (`validate.py`, `validate_part2.py`) and then
the extended check modules, each exposing `run(record, tmp, ROOT, KB)`:
`checks_msm.py` (Markov state models against deeptime, SciPy and analytical
chains), `checks_enhanced.py` (metadynamics and umbrella reweighting against
`plumed sum_hills`, pymbar and analytical potentials; DCCM, per-residue
differences, ligand contacts and dihedrals against NumPy, ProDy and MDAnalysis),
`checks_performance.py` (streaming engine, `ConformAtlasPCA`, HDF5 round trip,
`rebin`, `api`), `checks_report.py` (figure formats, interactive-figure data,
interpretive numbers) and `checks_docs.py` (tutorial commands and numbers,
benchmark table, DOIs, Sphinx `-W` build). Each module also runs standalone
(`python validation/checks_msm.py`). Every check is recorded with

```python
record(section, check, passed, measured, expected, tolerance, reference, note="")
```

Each record states what was measured, what the analytical or reference value is, the
tolerance that was applied (with its justification: counting noise, file precision,
bin width, binomial standard error of a trial count), and the reference
implementation or theorem. Checks that need an absent external dependency call
`record(..., note="skipped")` instead of failing.

### Adding a validation check

Every pull request that adds or changes a numerical result must add or update a
check. The pattern:

1. **Construct a system with a known answer.** Analytical where possible: a Gaussian
   well for a landscape (ΔG = ½ kT r²/σ², bin-integrated with `erf`), an AR(1) process
   for autocorrelation (g = (1+φ)/(1−φ)), a two-state Markov chain for populations,
   dwell times and MFPT, a double-well potential sampled by Metropolis Monte Carlo for
   basins and barriers. Where no closed form exists, use an independent implementation
   (MDAnalysis, pymbar, deeptime, SciPy, GROMACS, ProDy) on *identical input*.
2. **State the tolerance before looking at the result**, and derive it from the
   statistics of the test (counting noise kT/√(N p), binomial SE √(p(1−p)/n), file
   precision), not from the discrepancy you happen to observe.
3. **Put the check in a function** `check_<topic>()` in `validation/validate.py` (or
   `validate_part2.py`, or a new `validation/checks_<topic>.py` exposing
   `run(record, tmp, ROOT, KB)`), call `record(...)`, and import/call it from `main()`.
4. Run `python validation/validate.py`; commit the regenerated `VALIDATION.md` and
   `results.json`, then `python validation/checks_docs.py --write-benchmarks` to refresh
   `docs/benchmarks.md`.
5. Describe the check in the pull request and add a bullet under *Validated* in
   `docs/methods.md` §14 (or under *Heuristics* if it is a convention rather than a
   validated quantity).

## Pull requests

- One topic per pull request; include tests, validation and documentation changes
  with the code.
- Add a bullet to the *Unreleased* section of `CHANGELOG.md`
  ([Keep a Changelog](https://keepachangelog.com/) format).
- Fill in the pull-request template; the "validated vs heuristic" section is
  mandatory for numerical changes.
- CI must be green: ruff, pytest on Linux and macOS for Python 3.10–3.12, package
  build, and the Sphinx `-W` build.

## Reporting bugs and asking questions

Use the issue templates. A bug report that includes `run_metadata.json` (it records
versions, options and warnings) and, where possible, a trajectory fragment small
enough to attach, is far quicker to act on.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
