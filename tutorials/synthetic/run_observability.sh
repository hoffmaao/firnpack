#!/bin/bash
# OSSE observability ablation: nine subsets (co-located truth with dynamic
# strain; velocity "v" is a first-class observable; ezz free in every subset).
# Launch: nohup bash experiments/synthetic/run_observability.sh \
#           > test/southpole/logs/obsv_driver.log 2>&1 &
cd /home/andrew/projects/firngrain
export PYTHONPATH=src:experiments/synthetic OMP_NUM_THREADS=1
# 40 iters is under-converged for the history rows (knots move slowly);
# the matrix protocol is 120 (matches the archived ezz0 rerun).
export FIRN_MAX_ITER=${FIRN_MAX_ITER:-120}
PY=/home/andrew/venv-firedrake-2026/bin/python
L=test/southpole/logs
SUBSETS=("rho" "rho,v" "rho,age" "rho,age,dage" "rho,age,dage,v"
         "rho,age,dage,Tdeep" "rho,age,dage,Tdeep,Tsh"
         "rho,age,dage,Tdeep,Tsh,v" "Tdeep")

run_one () {
  local S=$1; local SLUG=${S//,/-}
  echo "[driver] $(date '+%T') start $S"
  FIRN_OBS_SUBSET=$S $PY experiments/synthetic/observability.py \
      > $L/obsv_${SLUG}.log 2>&1
  echo "[driver] $(date '+%T') done  $S (exit $?)"
}

for S in "${SUBSETS[@]}"; do
  run_one "$S" &
  while [ "$(jobs -r | wc -l)" -ge 9 ]; do wait -n; done
done
wait
echo "[driver] ALL_SUBSETS_DONE"
