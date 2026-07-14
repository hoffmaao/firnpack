#!/bin/bash
# Velocity-OSSE driver: all five cases concurrently (each single-process).
cd "$(dirname "$0")/../.."
export PYTHONPATH=src OMP_NUM_THREADS=1
PY=/home/andrew/venv-firedrake-2026/bin/python
L=experiments/synthetic/logs; mkdir -p $L
for C in nov vco vrho vmis vmis_sig; do
  echo "[vosse] $(date '+%T') start $C"
  FIRN_VCASE=$C $PY experiments/synthetic/velocity_osse.py > $L/vosse_$C.log 2>&1 &
done
wait
echo "VOSSE_DONE"
for C in nov vco vrho vmis vmis_sig; do
  echo "--- $C ---"; tail -12 $L/vosse_$C.log
done
