## Summary

<!-- What does this change do, and why? Link the issue if there is one. -->

## Type of change

- [ ] Bug fix
- [ ] New analysis / option / output
- [ ] Documentation only
- [ ] Refactoring (no change in results)

## Validation (mandatory for numerical changes)

<!-- Which analytical result or independent implementation is the new/changed quantity
     checked against? Name the check added to validation/validate.py (or
     validation/checks_<topic>.py), its tolerance and how the tolerance was derived. -->

| Quantity | Reference | Tolerance | Check function |
|---|---|---|---|
|  |  |  |  |

**Validated vs heuristic.** <!-- State explicitly which parts are validated to the
standard above and which are conventions/heuristics; the latter must be labelled as
such in docstrings and in docs/methods.md §14. -->

## Assumptions

<!-- Equal-weight replicates, lag-time Markovianity, reversibility, fixed basin
     labels, normality across replicates, … — anything the new code assumes. -->

## Checklist

- [ ] `ruff check src tests validation` and `ruff format --check src tests validation` pass
- [ ] `pytest -q` passes; new tests added under `tests/unit` or `tests/integration`
- [ ] `python validation/validate.py` passes; `VALIDATION.md` / `results.json` regenerated if a check was added
- [ ] `python -m sphinx -W -b html docs docs/_build/html` passes; docs updated (methods, outputs, CLI help, tutorial if numbers changed)
- [ ] `CHANGELOG.md` *Unreleased* entry added
- [ ] Units are Å / ps / kJ/mol and `R_GAS_CONSTANT` is used for kT
- [ ] No placeholders, mocks or TODO stubs in shipped code
