# HIV-1 protease Markov-model benchmark

Criteria: [`../CRITERIA_T4L_HIV.md`](../CRITERIA_T4L_HIV.md). Results: [`../RESULTS_T4L_HIV.md`](../RESULTS_T4L_HIV.md).
Data: Zenodo 10.5281/zenodo.5006613 (Sadiq, Muñiz Chicharro, Friedrich & Wade 2021, CC-BY 4.0); unpack
`data_MD_MSM_analysis.tar` and `data_MD_MSMobj.tar` under `$CA_BENCH_DATA/hiv1_protease/`.

```bash
python benchmarks/hiv/h_export_pyemma.py           # environment with PyEMMA 2.5 (Python 3.10)
python benchmarks/hiv/h_export_pyemma.py --hmm     # environment with PyEMMA + bhmm, Python 3.9, numpy 1.23 (the saved HMM)
CA_SRC=<timescale-fix src> python benchmarks/hiv/h_compare.py
```
The exported arrays (`results/*.npz`) are derived from the authors' data and are not committed.
