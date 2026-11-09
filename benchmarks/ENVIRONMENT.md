# Benchmark environment

Written by `benchmarks/adk/perf_suite.py` on 2026-11-07 23:53 IST.

| Item | Value |
|---|---|
| Python | 3.11.17 |
| numpy / scipy | 2.4.6 / 1.17.1 |
| MDAnalysis | 2.10.0 |
| ProDy | 2.6.1 |
| MDTraj | 1.11.1 |
| scikit-learn | 1.9.1 |
| deeptime | 0.4.5 |
| GROMACS | GROMACS version:     2026.3-conda_forge |
| ConformAtlas repository commit (this branch) | a105e8f |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700 |
| Logical CPUs visible to the benchmark | 20 (BLAS/OpenMP threads fixed to 8) |
| Platform | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 |

Notes: other jobs (a large background download) were running on the machine during the performance runs; the
load average before every run is recorded in `results/performance_runs.csv`. Peak memory is the maximum
resident set size reported by `/usr/bin/time` for the process; for GROMACS it is the largest `gmx` process (the Python parent that converts the trajectory is excluded).
