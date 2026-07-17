"""observability.py — OSSE observation-subset ablation (single subset per run).

Answers: WHICH observations make the record (T-history, accumulation) and the
firn parameters recoverable? Runs one cold-start inversion with the observation
subset given by env FIRN_OBS_SUBSET (comma list from {rho,age,dage,Tdeep,Tsh,v}),
identical truth + noise realization across subsets.

ALL observations are CO-LOCATED with the column (Andrew, 2026-07-12) — the
truth carries dynamic strain (ezz = -4e-5/yr) so the velocity observable "v"
(ApRES-like differenced profile, same dRdt_diff operator as South Pole) is
meaningful, and ezz_yr is a free control in EVERY subset (prior 0 +/- 1e-4):
subsets without v demonstrate ezz is unobservable from the core system.
(Non-co-located transfer experiments live in velocity_osse.py, not here.)

Run one subset:
  PYTHONPATH=src OMP_NUM_THREADS=1 \
  FIRN_OBS_SUBSET=rho,age <venv> tutorials/synthetic/diagnostics/observability.py

The paper's ablation is these nine subsets, each run on its own:
  rho
  rho,v
  rho,age
  rho,age,dage
  rho,age,dage,v
  rho,age,dage,Tdeep
  rho,age,dage,Tdeep,Tsh
  rho,age,dage,Tdeep,Tsh,v
  Tdeep
Output: output/obsv_<slug>.json (+ shared output/obsv_truth_profiles.json)
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from firnpack.inverse.osse import (TRUTH, T_YEARS, T_TRUTH, B_YEARS, B_TRUTH,
                                   truth_profiles, make_obs, inversion_cfg)
from firnpack.inverse import assimilate

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not diagnostics/
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
SUBSET = [s for s in os.environ.get("FIRN_OBS_SUBSET", "rho").split(",") if s]
SLUG = "-".join(SUBSET)
SPIN, DT, NZ = 1600.0, 5.0, 100
MAXIT = int(os.environ.get("FIRN_MAX_ITER", "40"))
EZZ_TRUTH = -4.0e-5

print(f"OBSERVABILITY subset = {SUBSET} (co-located truth, ezz={EZZ_TRUTH:+.1e})")
d, rho_t, age_t, T_t, w_t = truth_profiles(str(OUT), spin=SPIN, dt=DT, NZ=NZ,
                                           ezz_truth=EZZ_TRUTH, return_w=True)
tp = OUT / "obsv_truth_profiles.json"
if not tp.exists():
    json.dump(dict(depth=d.tolist(), rho=rho_t.tolist(), age_yr=age_t.tolist(),
                   T_C=T_t.tolist(), w_m_yr=w_t.tolist(),
                   truth=dict(TRUTH, ezz_yr=EZZ_TRUTH),
                   T_years=T_YEARS.tolist(), T_truth=T_TRUTH.tolist(),
                   b_years=B_YEARS.tolist(), b_truth=B_TRUTH.tolist(),
                   spin=SPIN, dt=DT, NZ=NZ), open(tp, "w"), indent=1)
obs = make_obs(d, rho_t, age_t, T_t, SUBSET, seed=0, w_t=w_t)
cfg = inversion_cfg(obs, str(OUT), f"obsv_{SLUG}", spin=SPIN, dt=DT, NZ=NZ,
                    maxit=MAXIT, with_ezz=True)
r = assimilate(cfg, mode="optimize")
truth_all = dict(TRUTH, ezz_yr=EZZ_TRUTH)
print(f"\nsubset {SUBSET}: J={r['J']:.2f}")
for k, tv in truth_all.items():
    if k in r["m_map"]:
        print(f"  {k:12s} truth={tv:.4g} rec={r['m_map'][k]:.4g} err={r['m_map'][k]-tv:+.3g}")
