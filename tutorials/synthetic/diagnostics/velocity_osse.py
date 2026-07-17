"""velocity_osse.py — vertical-velocity (ApRES-like) OSSE: what does velocity
data add, and what does adopting a NON-CO-LOCATED profile cost?

NOTE (2026-07-12): the PAPER OSSE assumes all observations co-located
(Andrew) — that lives in observability.py (v is a subset member there, same
truth/operator). This script remains as the SP-motivated transfer SIDE-STUDY
(vmis/vmis_sig); its nov/vco/vrho results were promoted into the co-located
observability matrix (see output/*provenance fields).

Motivated by the South Pole ApRES verdict: the core observations (rho, age,
d(age)/dz, T) come from the ice-core site, but the velocity profiles come from
array sites kilometres away in a spatially variable strain field. We adopted a
screened remote profile there ("held-out consistency" selection); here the
same protocol is tested under KNOWN truth.

All cases share one core-column truth (ezz = -4e-5/yr, the kinematic-scale
value) and the same noise realizations. Cases (env FIRN_VCASE):

  nov       core obs only, ezz free                — is ezz observable at all
                                                     without velocity?
  vco       + v co-located (same column)           — the value of velocity
  vrho      rho + v only                           — what velocity alone (with
                                                     the density column) carries
  vmis      + v from a companion column with       — TRANSFER BIAS: assimilate
            ezz = -7e-5 (spatial strain             a mismatched remote profile
            variability; the steep-regime case),    as if co-located, stated
            sigma_v = 0.012 as stated               sigma
  vmis_sig  same, sigma_v inflated to 0.030        — does an honest transfer
            (cross-regime spread)                    systematic contain the bias?

Key outputs per case: recovered ezz vs truth, and the bias induced in the
densification-law parameters / histories relative to vco.

Run one case (FIRN_VCASE in {nov, vco, vrho, vmis, vmis_sig}):
  PYTHONPATH=src OMP_NUM_THREADS=1 FIRN_VCASE=vmis \
    <venv-python> tutorials/synthetic/diagnostics/velocity_osse.py
Output: output/vosse_<case>.json.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from firnpack.inverse.osse import TRUTH, truth_profiles, make_obs, inversion_cfg
from firnpack.inverse import assimilate

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not diagnostics/
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
CASE = os.environ.get("FIRN_VCASE", "nov")
SPIN, DT, NZ = 1600.0, 5.0, 100
MAXIT = int(os.environ.get("FIRN_MAX_ITER", "40"))
EZZ_CORE, EZZ_MIS = -4.0e-5, -7.0e-5
V_SIG = {"vmis_sig": 0.030}.get(CASE, 0.012)

print(f"VELOCITY-OSSE case={CASE} (core-column truth ezz={EZZ_CORE:+.1e}/yr, "
      f"v sigma={V_SIG})")
d, rho_t, age_t, T_t, w_t = truth_profiles(str(OUT), spin=SPIN, dt=DT, NZ=NZ,
                                           ezz_truth=EZZ_CORE, return_w=True)
tp = OUT / "vosse_truth.json"
if not tp.exists():
    json.dump(dict(depth=d.tolist(), rho=rho_t.tolist(), age_yr=age_t.tolist(),
                   T_C=T_t.tolist(), w_m_yr=w_t.tolist(),
                   truth=dict(TRUTH, ezz_yr=EZZ_CORE), ezz_mis=EZZ_MIS,
                   spin=SPIN, dt=DT, NZ=NZ), open(tp, "w"), indent=1)

CORE = ["rho", "age", "dage", "Tdeep", "Tsh"]
WHICH = {"nov": CORE, "vco": CORE + ["v"], "vrho": ["rho", "v"],
         "vmis": CORE + ["v"], "vmis_sig": CORE + ["v"]}[CASE]

if CASE in ("vmis", "vmis_sig"):
    # companion column: identical law/histories, different strain regime
    dm, _, _, _, w_m = truth_profiles(str(OUT), spin=SPIN, dt=DT, NZ=NZ,
                                      ezz_truth=EZZ_MIS, return_w=True)
    w_for_v = np.interp(d, dm, w_m)
else:
    w_for_v = w_t

obs = make_obs(d, rho_t, age_t, T_t, WHICH, seed=0, w_t=w_for_v, v_sig=V_SIG)
cfg = inversion_cfg(obs, str(OUT), f"vosse_{CASE}", spin=SPIN, dt=DT, NZ=NZ,
                    maxit=MAXIT, with_ezz=True)
r = assimilate(cfg, mode="optimize")

truth_all = dict(TRUTH, ezz_yr=EZZ_CORE)
print(f"\ncase {CASE}: J={r['J']:.2f}")
for k, tv in truth_all.items():
    if k in r["m_map"]:
        rec = r["m_map"][k]
        print(f"  {k:12s} truth={tv:.4g} rec={rec:.4g} err={rec - tv:+.3g}")
