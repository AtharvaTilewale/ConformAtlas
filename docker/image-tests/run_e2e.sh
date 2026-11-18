#!/usr/bin/env bash
# End-to-end tests of the ConformAtlas command line inside the container image.
#
# Usage (inside the image):  bash run_e2e.sh SRC_DIR DATA_DIR WORK_DIR MDA_DIR
#   SRC_DIR   checkout of the release under test (provides tests/data/h1n1_ca)
#   DATA_DIR  output of make_test_data.py
#   WORK_DIR  writable directory for results
#   MDA_DIR   MDAnalysisTests data (adk_oplsaa.tpr, adk_oplsaa.xtc)
#
# Every test is run even if an earlier one fails; the script exits non-zero if
# any test failed. "graceful" tests check that optional features missing from
# the image (GROMACS backend, PDF report) are reported without a Python traceback.
# "known" tests reproduce open ConformAtlas bugs: they are reported as KNOWN (or
# FIXED once they pass) and do not fail the run.

set -u
SRC=$1; DATA=$2; WORK=$3; MDA=$4
H1=$SRC/tests/data/h1n1_ca
mkdir -p "$WORK"; cd "$WORK" || exit 2
export MPLBACKEND=Agg
PASS=0; FAIL=0; KNOWN=0; RESULTS=()

run() {  # run NAME EXPECT(ok|graceful|known) CHECK_FILE -- command...
  local name=$1 expect=$2 check=$3; shift 4
  local log="$WORK/$name.log" start end rc status
  start=$(date +%s)
  "$@" >"$log" 2>&1; rc=$?
  end=$(date +%s)
  if [ "$expect" = ok ] || [ "$expect" = known ]; then
    if [ $rc -eq 0 ] && { [ "$check" = - ] || [ -e "$check" ]; }; then status=PASS; else status=FAIL; fi
    if [ "$expect" = known ]; then
      if [ $status = PASS ]; then status=FIXED; else status=KNOWN; fi
    fi
  else
    # any exit code is fine as long as there is no Python traceback
    if ! grep -q "^Traceback" "$log"; then status=PASS; else status=FAIL; fi
  fi
  case $status in
    PASS|FIXED) PASS=$((PASS + 1)) ;;
    KNOWN) KNOWN=$((KNOWN + 1)) ;;
    *) FAIL=$((FAIL + 1)) ;;
  esac
  RESULTS+=("$(printf '%-5s %-34s rc=%-3s %4ss' "$status" "$name" "$rc" "$((end - start))")")
  echo "${RESULTS[-1]}"
  if [ $status != PASS ] || [ "$expect" = graceful ]; then sed 's/^/      | /' "$log" | tail -8; fi
}

ADK="-s $DATA/adk.pdb -f $DATA/adk_dims.xtc -T 300 -q --no-progress"
NA="-s $H1/neuraminidase_ca.pdb -f $H1/neuraminidase_ca.xtc -n $H1/index.ndx -T 300 -q --no-progress"

# --- real protein-ligand trajectory (neuraminidase-zanamivir, 81 frames) with index file
run real_ligand_residue ok out_na/residues -- \
  conformatlas analyze $NA --cov-group C-alpha --lsq-group C-alpha --lig ZMR \
  -b 5000 --bootstrap 50 --residue-analysis -o out_na
run real_outputs_complete ok - -- \
  sh -c 'for p in report.html summary.json pca fel states statistics kinetics convergence residues figures; do test -e out_na/$p || { echo missing $p; exit 1; }; done'
run real_msm ok out_na_msm/msm -- \
  conformatlas analyze $NA --cov-group C-alpha --lsq-group C-alpha --msm --msm-lag 5 -o out_na_msm

# --- GROMACS run input (.tpr) with a solvated system
run tpr_input ok out_tpr/report.html -- \
  conformatlas analyze -s $MDA/adk_oplsaa.tpr -f $MDA/adk_oplsaa.xtc -T 300 -q --no-progress \
  --bins 10 --n-components 3 -o out_tpr

# --- decompositions and collective variables on full-atom AdK
run dihedral_backbone ok out_dih/report.html -- conformatlas analyze $ADK --pca-type dihedral --dihedral-set backbone -o out_dih
run dihedral_backbone_chi ok out_dihchi/report.html -- conformatlas analyze $ADK --pca-type dihedral --dihedral-set backbone+chi -o out_dihchi
run tica ok out_tica/report.html -- conformatlas analyze $ADK --pca-type tica --tica-lag 5 -o out_tica
run rmsd_rg_cv ok out_rg/report.html -- conformatlas analyze $ADK --cv rmsd-rg -o out_rg
run streaming_float32_workers ok out_stream/report.html -- \
  conformatlas analyze $ADK --engine streaming --precision float32 --n-workers 2 --chunk-size 20 -o out_stream
run msm_regular_space ok out_msmrs/msm -- \
  conformatlas analyze $ADK --msm --msm-lag 2 --msm-clustering regular-space --msm-dmin 1.0 -o out_msmrs
run hdf5_and_reanalysis ok out_from/report.html -- \
  sh -c "conformatlas analyze $ADK --save-hdf5 -o out_h5 && test -s out_h5/results.h5 && conformatlas analyze --from out_h5 -q --no-progress --bins 20 -o out_from"
run all_figure_formats ok out_fig/figures -- conformatlas analyze $ADK --fig-format all -o out_fig

# --- biased sampling: PLUMED metadynamics and umbrella sampling
run reweight_metad_tp ok rw_tp/pmf.csv -- \
  conformatlas reweight metad --hills $DATA/HILLS --colvar $DATA/COLVAR --cv x -T 300 -o rw_tp
run reweight_metad_final_bias ok rw_fb/pmf.csv -- \
  conformatlas reweight metad --hills $DATA/HILLS --colvar $DATA/COLVAR --cv x -T 300 --method final-bias -o rw_fb
CENTERS=$(cat "$DATA/centers.txt")
UMB_ARGS=""; for c in $CENTERS; do UMB_ARGS="$UMB_ARGS --center $c"; done
for f in "$DATA"/COLVAR_*; do UMB_ARGS="$UMB_ARGS --colvar $f"; done
run reweight_umbrella_mbar ok rw_mbar/pmf.csv -- \
  conformatlas reweight umbrella $UMB_ARGS --cv d --kappa 250 -T 300 --method mbar --skip-time 100 -o rw_mbar
run reweight_umbrella_wham ok rw_wham/pmf.csv -- \
  conformatlas reweight umbrella $UMB_ARGS --cv d --kappa 250 -T 300 --method wham --skip-time 100 -o rw_wham
run weights_file ok out_w/report.html -- \
  conformatlas analyze $NA --cov-group C-alpha --lsq-group C-alpha --weights-file $DATA/weights_uniform.dat -o out_w

# --- comparison and configuration files
run example_compare_resegment ok cmp/comparison_interactive.html -- \
  sh -c "conformatlas example -o ex >/dev/null && conformatlas compare -c ex/comparison.yaml --bootstrap-mode resegment -o cmp"
run init_config_roundtrip ok cfg_out/report.html -- \
  sh -c "conformatlas init-config --type analyze -o run.yaml && conformatlas analyze --config run.yaml $ADK -o cfg_out"

# --- known ConformAtlas 1.0.2 bugs (reported, not counted as failures)
# Ligand contacts ignore the frames discarded by --equilibration auto
# ("Contact analysis produced N frames but M frames are assigned").
run known_contacts_with_auto_equilibration known out_known1/residues -- \
  conformatlas analyze $NA --cov-group C-alpha --lsq-group C-alpha --lig ZMR \
  --equilibration auto --residue-analysis -o out_known1
# When no lag in the implied-timescale scan gives a valid MSM, the run stops with
# KeyError: 'lag_frames' instead of a clear message (expected: graceful error).
run known_msm_no_valid_lag_message known - -- \
  sh -c "conformatlas analyze $ADK --msm --msm-lag 2 -o out_known2 > known2.out 2>&1; ! grep -q lag_frames known2.out"

# --- optional features that are not in the image must be reported cleanly
run graceful_gromacs_backend graceful - -- conformatlas analyze $ADK --pca-backend gromacs -o out_gmx
run graceful_pdf_report graceful - -- conformatlas analyze $ADK --pdf -o out_pdf

echo
echo "================ summary: $PASS passed, $FAIL failed, $KNOWN known issues ================"
printf '%s\n' "${RESULTS[@]}"
[ $FAIL -eq 0 ]
