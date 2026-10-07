"""temporal_resolution.py - does denser temporal sampling of T(t)/b(t) help?

The OSSE arbiter for the knot-density question: a truth WITH decadal structure
(AR(1) anomalies: b ±5% log, T ±0.3 °C post-1800) is inverted twice from the
same full observation set + noise:

  FIRN_TR_CASE=coarse : the standard layouts (6 T / 8 b knots)
  FIRN_TR_CASE=dense  : 25 T (10-yr post-1800) / 54 b (25-yr, 10-yr post-1800)

With truth known, "recovers finer structure" vs "invents noise" is measurable:
compare recovered-vs-truth rms in multidecadal and decadal bands per case.

Run one case (FIRN_TR_CASE in {coarse, dense}):
  PYTHONPATH=src OMP_NUM_THREADS=1 FIRN_TR_CASE=dense \
  <venv> tutorials/synthetic/diagnostics/temporal_resolution.py
Output: output/tr_truth.json, output/tr_<case>.json
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from firnpack.inverse.osse import (dense_truth, truth_profiles, make_obs,
                                   inversion_cfg, TRUTH)
from firnpack.inverse import assimilate

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not diagnostics/
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
CASE = os.environ.get("FIRN_TR_CASE", "coarse")
SPIN, DT, NZ = 1600.0, 5.0, 100
MAXIT = int(os.environ.get("FIRN_MAX_ITER", "120"))

Ty_t, Tv_t, By_t, Bv_t = dense_truth(seed=7)
print(f"TEMPORAL-RESOLUTION case={CASE}: truth has {len(Ty_t)} T / {len(By_t)} b knots "
      f"(decadal structure); inverting with "
      f"{'standard coarse' if CASE=='coarse' else 'dense'} layouts")
d, rho_t, age_t, T_t = truth_profiles(str(OUT), spin=SPIN, dt=DT, NZ=NZ,
                                      T_years=Ty_t, T_vals=Tv_t,
                                      B_years=By_t, B_vals=Bv_t)
tp = OUT / "tr_truth.json"
if not tp.exists():
    json.dump(dict(T_years=Ty_t.tolist(), T_truth=Tv_t.tolist(),
                   B_years=By_t.tolist(), B_truth=Bv_t.tolist(), truth=TRUTH,
                   depth=d.tolist(), rho=rho_t.tolist(), age_yr=age_t.tolist(),
                   T_C=T_t.tolist(), seed=7, spin=SPIN, dt=DT), open(tp, "w"), indent=1)

obs = make_obs(d, rho_t, age_t, T_t, ["rho", "age", "dage", "Tdeep", "Tsh"], seed=0)
if CASE == "dense":
    Ty_i = np.concatenate([np.array([1000., 1300., 1600.]), np.arange(1800., 2016., 10.)])
    By_i = np.concatenate([np.arange(1000., 1800., 25.), np.arange(1800., 2016., 10.)])
else:
    Ty_i, By_i = None, None   # standard coarse layouts
cfg = inversion_cfg(obs, str(OUT), f"tr_{CASE}", spin=SPIN, dt=DT, NZ=NZ,
                    maxit=MAXIT, T_years=Ty_i, B_years=By_i)
r = assimilate(cfg, mode="optimize")
print(f"\ncase {CASE}: J={r['J']:.2f}, n_ctrl={r['n_ctrl']}")
