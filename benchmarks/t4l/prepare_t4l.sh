#!/bin/bash
# Prepare T4L benchmark inputs: protein-only run file per force field and a 100 ps copy of every trajectory.
# Usage: prepare_t4l.sh <data root containing t4l/>   (needs GROMACS `gmx` on PATH or $GMX)
set -euo pipefail
ROOT=${1:?data root}; GMX=${GMX:-gmx}; export GMX_MAXBACKUP=-1
for d in "$ROOT"/t4l/ff99SBstar-ILDN_TIP4P2005_5x1us "$ROOT"/t4l/ff15ipq_SPCEb_3x1us; do
  out="$d/prepared"; mkdir -p "$out"
  first=$(ls "$d"/*.tpr | head -1)
  echo Protein | "$GMX" convert-tpr -s "$first" -o "$out/protein.tpr" > "$out/convert-tpr.log" 2>&1
  for x in "$d"/*.xtc; do
    name=$(basename "$x" .xtc)
    [ -s "$out/${name}_100ps.xtc" ] && { echo "skip $name"; continue; }
    "$GMX" trjconv -f "$x" -o "$out/${name}_100ps.tmp.xtc" -dt 100 > "$out/${name}_trjconv.log" 2>&1
    mv "$out/${name}_100ps.tmp.xtc" "$out/${name}_100ps.xtc"
    echo "$(date +%H:%M:%S) done $name"
  done
done
