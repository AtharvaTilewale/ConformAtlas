# ADK equilibrium trajectory benchmark

Pass criteria: [`../CRITERIA.md`](../CRITERIA.md) (committed before any result). Deviations found while running:
[`../DEVIATIONS.md`](../DEVIATIONS.md). Environment: [`../ENVIRONMENT.md`](../ENVIRONMENT.md). Results:
[`RESULTS.md`](RESULTS.md), generated from `results/*.json`.

Data are public and are not stored in this repository. Fetch them (resumable, md5-verified) and run in this order:

```bash
export CA_BENCH_DATA=/path/to/data            # default: benchmarks/data
export GMX=/path/to/gmx                        # default: gmx on PATH (GROMACS for G1, G2, G5, timings)
python benchmarks/download_data.py             # ADK files; also queues the HIV-1 protease and T4L sets
python benchmarks/adk/g1_pca_numerics.py       # G0 integrity, G1 PCA numerics
python benchmarks/adk/g2_fel.py                # G2 free-energy landscape vs NumPy and gmx sham
python benchmarks/adk/g3_angles_pmf.py         # G3 angle coordinates and the published PMF
python benchmarks/adk/g4_kinetics.py           # G4 TICA / MSM vs deeptime
python benchmarks/adk/g5_default_workflows.py  # G5 each tool's default workflow, RMSIP
python benchmarks/adk/g6_informational.py      # crystal structures in PC space, default analyze run
python benchmarks/adk/perf_suite.py            # timings and peak memory (reported, not gated)
python benchmarks/adk/build_report.py          # RESULTS.md
```

Sources: ADK equilibrium trajectory, figshare 10.6084/m9.figshare.5108170 (Seyler & Beckstein 2017, CC-BY 4.0);
apo PMF, Zenodo 10.5281/zenodo.29617 (Beckstein et al. 2009, CC-BY-SA 4.0); structures 4AKE and 1AKE, RCSB PDB.
Tools: MDAnalysis, ProDy, MDTraj with scikit-learn, GROMACS and deeptime (versions in `ENVIRONMENT.md`).
