# T4 lysozyme replicate benchmark

Criteria: [`../CRITERIA_T4L_HIV.md`](../CRITERIA_T4L_HIV.md). Results: [`../RESULTS_T4L_HIV.md`](../RESULTS_T4L_HIV.md).
Data: Zenodo 10.5281/zenodo.3989057 and 10.5281/zenodo.3991429 (fetched by `../download_data.py`; not stored here).

```bash
export CA_BENCH_DATA=/path/to/data
GMX=gmx GMX_CONVERT=/path/to/gromacs-2023/gmx bash benchmarks/t4l/prepare_t4l.sh "$CA_BENCH_DATA"   # whole-molecule 100 ps copies
CA_SRC=src python benchmarks/t4l/t1_t2_pca_fel.py          # T0 integrity, T1 PCA numerics, T2 landscape
python benchmarks/t4l/t4_null_test.py                       # T4 null test of `conformatlas compare`
python benchmarks/t4l/t4_null_test.py --force-field         # T5 force-field comparison
CA_SRC=<timescale-fix src> python benchmarks/t4l/t3_kinetics.py   # T3 TICA / MSM vs deeptime
python benchmarks/t4l/t6_replicates.py                      # T6 replicate agreement
python benchmarks/t4l/perf_suite_t4l.py --perf-src <perf-branch src>   # T7 timings
python benchmarks/build_report_t4l_hiv.py
```
The run file used by MDAnalysis must be written by a GROMACS release whose tpx format MDAnalysis can read
(2023.3 here; see DEVIATIONS D8).
