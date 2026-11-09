#!/bin/bash
# Prepare T4L benchmark inputs: protein-only run file per force field and a whole-molecule 100 ps copy of every trajectory.
# Usage: prepare_t4l.sh <data root containing t4l/>   (needs GROMACS `gmx` on PATH or $GMX)
set -euo pipefail
ROOT=${1:?data root}; GMX=${GMX:-gmx}; export GMX_MAXBACKUP=-1
# The protein-only run file must be readable by MDAnalysis: write it with a GROMACS release whose
# tpx format MDAnalysis supports (2023.3 here; 2026.3 writes tpx 138, unsupported by MDAnalysis 2.10).
GMX_CONVERT=${GMX_CONVERT:-$GMX}
for d in "$ROOT"/t4l/ff99SBstar-ILDN_TIP4P2005_5x1us "$ROOT"/t4l/ff15ipq_SPCEb_3x1us; do
  out="$d/prepared"; mkdir -p "$out"
  first=$(ls "$d"/*.tpr | head -1)
  [ -s "$out/protein.tpr" ] || echo Protein | "$GMX_CONVERT" convert-tpr -s "$first" -o "$out/protein.tpr" > "$out/convert-tpr.log" 2>&1
  for x in "$d"/*.xtc; do
    name=$(basename "$x" .xtc)
    [ -s "$out/${name}_100ps_whole.xtc" ] && { echo "skip $name"; continue; }
    # The deposited protein-only trajectories are not made whole across the periodic boundary in every
    # frame; -pbc whole repairs that (bonds from the protein-only run file).
    echo 0 | "$GMX" trjconv -f "$x" -s "$out/protein.tpr" -o "$out/${name}_100ps_whole.tmp.xtc" -dt 100 -pbc whole \
      > "$out/${name}_trjconv.log" 2>&1
    mv "$out/${name}_100ps_whole.tmp.xtc" "$out/${name}_100ps_whole.xtc"
    echo "$(date +%H:%M:%S) done $name"
  done
done
