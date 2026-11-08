# Adenylate kinase (ADK) example data

Reduced copy of a published equilibrium simulation of apo *E. coli* adenylate kinase,
used by the [tutorial](../../docs/tutorial_adk.md).

**Source.** S. L. Seyler and O. Beckstein, *Molecular dynamics trajectory for
benchmarking MDAnalysis*, figshare (2017),
[doi:10.6084/m9.figshare.5108170](https://doi.org/10.6084/m9.figshare.5108170),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Deposit
description: CHARMM27, explicit water and ions, NPT at 300 K and 1 bar, 1.004 µs on
Anton, frames every 240 ps, protein only, superposed on the CORE domain. Crystal
structures 1AKE and 4AKE: RCSB Protein Data Bank (public domain).

**Changes made here.** Only the Cα atoms are kept. The trajectory is converted to XTC
(maximum coordinate change 0.005 Å). The domain angles are computed from the backbone
of the full-atom trajectory. Everything is produced by
[`make_adk_data.py`](make_adk_data.py), which checks the md5 sums of the downloaded
files:

```bash
python examples/adk/make_adk_data.py --download /tmp/adk_source
```

| File | Content |
|---|---|
| `adk_ca.pdb`, `adk_ca.xtc` | 214 Cα atoms, 4,187 frames, time in ps |
| `index.ndx` | `System`, `C-alpha`, `CORE`, `NMP` (residues 30–59), `LID` (122–159) |
| `domain_angles.dat` | time (ps), θNMP, θLID (degrees); Beckstein *et al.* (2009) definitions |
| `crystal/*_A_ca.pdb` | Cα atoms of chain A of 1AKE (closed) and 4AKE (open) |
| `crystal/crystal_angles.json` | θNMP, θLID of the two crystal structures |

If you use these data, cite the deposit above and Seyler, Kumar, Thorpe & Beckstein,
*PLoS Comput. Biol.* 11 (2015) e1004568.

The same script writes the small regression fixture `tests/data/adk_ca/`. It contains
every 40th frame, the Cα atoms plus an Ap5A ligand whose coordinates are
**synthetic** (placed from 1AKE by superposition on the CORE domain, with noise), and
the heavy atoms of 4AKE chain A.
