#!/bin/bash
# Round-5 optimization (Calonne k + seam offset + d(age)/dz; deep domain
# activates automatically if FIRN_DEEP_T_FILE is exported before launch).
# Launch: nohup bash test/southpole/scripts/assimilation/run_r5_chain.sh \
#            > test/southpole/logs/r5_chain.log 2>&1 &
set -e
cd /home/andrew/projects/firngrain
export PYTHONPATH=src OMP_NUM_THREADS=1
PY=/home/andrew/venv-firedrake-2026/bin/python
L=test/southpole/logs
TAG=${FIRN_TAG:-sp_joint_r5}

echo "[chain] $(date '+%F %T') $TAG optimize (FIRN_MAX_ITER=${FIRN_MAX_ITER:-60}, T_SHIFT=${FIRN_T_SHIFT:-0}, KSNOW=${FIRN_KSNOW_FIX:-free}, DEEP=${FIRN_DEEP_T_FILE:-none})"
FIRN_MAX_ITER=${FIRN_MAX_ITER:-60} $PY test/southpole/scripts/assimilation/sp_joint_assimilate_r5.py \
    > $L/${TAG}_opt.log 2>&1

echo "[chain] $(date '+%F %T') diagnostics at $TAG MAP"
FIRN_MAP=${TAG}.json $PY test/southpole/scripts/diagnostics/sp_midrecord_diagnostic.py \
    > $L/${TAG}_diagnostic.log 2>&1 || echo "[chain] diagnostic FAILED (non-fatal)"

echo "[chain] $(date '+%F %T') CHAIN_COMPLETE"
