# T4 lysozyme replicate benchmark

Criteria: [`../CRITERIA_T4L_HIV.md`](../CRITERIA_T4L_HIV.md). Results: [`../RESULTS_T4L_HIV.md`](../RESULTS_T4L_HIV.md).
Data: Zenodo 10.5281/zenodo.3989057 and 10.5281/zenodo.3991429 (Hoffmann, Mulder & Schäfer). The records are open
access but state no licence, so the trajectories are not redistributed here; fetch them with `../download_data.py`.

```bash
export CA_BENCH_DATA=/path/to/data      # default: benchmarks/data
export CA_BENCH_WORK=/path/to/work      # analyze/compare runs of T4-T6 (default: $CA_BENCH_DATA/work)
GMX=gmx GMX_CONVERT=/path/to/gromacs-2023/gmx bash benchmarks/t4l/prepare_t4l.sh "$CA_BENCH_DATA"   # whole-molecule 100 ps copies
CA_SRC=src python benchmarks/t4l/t1_t2_pca_fel.py          # T0 integrity, T1 PCA numerics, T2 landscape
python benchmarks/t4l/t4_null_test.py                       # T4 null test of `conformatlas compare`
python benchmarks/t4l/t4_null_test.py --force-field         # T5 force-field comparison
python benchmarks/t4l/t4_null_test.py --tag _dev [--force-field]   # T4-R / T5 re-run with the dev code
CA_SRC=src python benchmarks/t4l/t3_kinetics.py   # T3 TICA / MSM vs deeptime (src of dev, any commit >= a1dac8e)
python benchmarks/t4l/t6_replicates.py                      # T6 replicate agreement
python benchmarks/t4l/perf_suite_t4l.py --perf-src src   # T7 timings (src of dev, any commit >= d6bcbc2, the truncated-PCA merge)
python benchmarks/build_report_t4l_hiv.py
```
The run file used by MDAnalysis must be written by a GROMACS release whose tpx format MDAnalysis can read
(2023.3 here; see DEVIATIONS D8).
