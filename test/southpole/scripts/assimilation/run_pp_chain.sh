#!/bin/bash
# Per-point (honest error model) round: re-opt -> Hessian -> sensitivities -> combine.
# Launch: nohup bash test/southpole/scripts/assimilation/run_pp_chain.sh \
#            > test/southpole/logs/pp_chain.log 2>&1 &
# Logs land in test/southpole/logs/ (repo-side, survives sessions).
set -e
cd /home/andrew/projects/firngrain
export PYTHONPATH=src OMP_NUM_THREADS=1
PY=/home/andrew/venv-firedrake-2026/bin/python
L=test/southpole/logs

echo "[chain] $(date '+%F %T') pp optimize (FIRN_MAX_ITER=${FIRN_MAX_ITER:-50})"
FIRN_MAX_ITER=${FIRN_MAX_ITER:-50} $PY test/southpole/scripts/assimilation/sp_joint_assimilate_pp.py \
    > $L/pp_opt.log 2>&1

echo "[chain] $(date '+%F %T') pp Hessian (93 gradient evals)"
$PY test/southpole/scripts/assimilation/sp_uq_hessian_pp.py > $L/pp_hessian.log 2>&1

echo "[chain] $(date '+%F %T') pp attribution sensitivities"
$PY test/southpole/scripts/reanalysis/sp_uq_sensitivity_pp.py > $L/pp_sensitivity.log 2>&1

echo "[chain] $(date '+%F %T') combine"
$PY test/southpole/scripts/reanalysis/sp_uq_combine_pp.py > $L/pp_combine.log 2>&1

echo "[chain] $(date '+%F %T') CHAIN_COMPLETE"
