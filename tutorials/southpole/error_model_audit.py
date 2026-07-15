"""error_model_audit.py — misfits against STATED/derived measurement errors.

The model predictions don't depend on sigma, so this re-normalizes the MAP
residuals per observation block against (a) the in-use error model and (b) the
strict stated/derived measurement errors:

  density  : 5.5 kg/m3       (empirical point scatter, obs-error analysis)
  age      : 3 + 0.5% yr     (layer-counting scale in the upper core)
  dage     : window-fit s.e. (no 4% floor)
  T        : 0.04 C          (empirical profile noise / sensor spec)
  velocity : ApRES stated v_unc_m_yr (median of the used bins; outlier-filtered)

Also reports the implied representation error per block,
  sigma_repr = sqrt(max(<r^2> - sigma_meas^2, 0)),
i.e. the sigma_repr a two-component model sigma^2 = sigma_meas^2 + sigma_repr^2
would need for chi2/N = 1 — measured from the data, not chosen.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/error_model_audit.py
Env: FIRN_MAP_JSON (default frozen r8), FIRN_KNOTS to match.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
MAP_PATH = os.environ.get("FIRN_MAP_JSON",
                          str(HERE/"results/sp_joint_r8.json"))
ns = {"__file__": str(HERE/"run.py"), "__name__": "cfgbuild"}
src = open(HERE/"run.py").read().split("warm = json.load")[0]
exec(src, ns)
warm = json.load(open(MAP_PATH))
from firnpack.inverse import assimilate
from firnpack.models.firn import FirnParameters
from firnpack.constants import year as YEAR_S
_p = FirnParameters(); C_I = float(_p.c_i)

r = assimilate(ns["cfg"], mode="forward", warm=warm, verbose=False)
blocks = {b["label"]: b for b in r["obs"]}

# ---- stated/derived measurement sigmas ----
def stated_sigma(lab, d, o):
    if lab == "rho":  return np.full(len(d), 5.5)                       # kg/m3
    if lab == "age":  return (3.0 + 0.005*(o/YEAR_S))*YEAR_S            # seconds
    if lab == "T":    return np.full(len(d), 0.04*C_I)                  # J/kg
    if lab == "dage":
        # recompute pure window-fit s.e. (no floor) at the block's centers
        a = pd.read_csv(HERE/"data/sp19_depth_age.csv"); a["age_yr"]=2015.0-a.year_CE
        a = a.query("depth_m<=130 and age_yr>=0")
        ad, aa = a.depth_m.values, a.age_yr.values
        ses = []
        for c in d:
            s = np.abs(ad-c) <= 0.6
            A = np.vstack([ad[s]-c, np.ones(s.sum())]).T
            coef, res, *_ = np.linalg.lstsq(A, aa[s], rcond=None)
            se = math.sqrt(max(float(res[0]) if len(res) else 0.0, 1e-12)/(s.sum()-2)
                           / max(np.sum((ad[s]-c)**2), 1e-12))
            ses.append(max(se, 1e-3))
        return np.array(ses)
    if lab.startswith("v"):   # any per-site velocity block (v, v_x17s2, ...)
        if os.environ.get("FIRN_VEL_SRC", "authors") == "zeising":
            # The zeising builder sets sig^2 = ee^2 + eref^2 + sig_shape^2, so
            # the pure MEASUREMENT part is the in-use sigma with the cross-site
            # shape systematic removed. (Do NOT use pipeline-A's v_unc_m_yr
            # here: different product, and its tail is pathological -> 1.3e6.)
            SIG_SHAPE = 3.5e-3
            s_use = np.array(blocks[lab]["sig"])
            return np.sqrt(np.maximum(s_use**2 - SIG_SHAPE**2, (1e-6)**2))
        ap = pd.read_csv(HERE/"data/apres_vertical_velocity_processed.csv")
        ap = ap[(ap.range_m<=130) & ap.v_smooth_m_yr.notna() & (ap.coherence>0.5)]
        u = ap.v_unc_m_yr.values; u = u[np.isfinite(u) & (u < 0.1)]
        return np.full(len(d), max(float(np.median(u)), 1e-4))
    raise ValueError(lab)

print(f"MAP: {Path(MAP_PATH).stem}   (forward J = {r['J']:.2f} under in-use sigmas)")
print(f"{'block':6s} {'rms(in-use σ)':>14s} {'rms(stated σ)':>14s} {'σ_meas':>12s} {'σ_repr needed':>14s}")
summary = {}
for lab in [l for l in ["rho", "age", "dage", "T"] if l in blocks] \
           + sorted(k for k in blocks if k.startswith("v")):
    b = blocks[lab]
    d = np.array(b["depths"]); o = np.array(b["obs"]); p = np.array(b["pred"])
    s_use = np.array(b["sig"]); s_st = stated_sigma(lab, d, o)
    res = p - o
    rms_use = float(np.sqrt(np.mean((res/s_use)**2)))
    rms_st = float(np.sqrt(np.mean((res/s_st)**2)))
    # representation sigma for chi2/N=1 (in physical units, block-median scale)
    r2 = np.mean(res**2); sm2 = np.mean(s_st**2)
    s_repr = math.sqrt(max(r2 - sm2, 0.0))
    # display units
    # multi-site velocity labels are v_<site>; they all convert as "v"
    conv = dict(rho=(1, "kg/m3"), age=(1/YEAR_S, "yr"), dage=(1, "yr/m"),
                T=(1/C_I, "C"), v=(1, "m/yr"))["v" if lab.startswith("v") else lab]
    print(f"{lab:6s} {rms_use:14.2f} {rms_st:14.2f} "
          f"{np.median(s_st)*conv[0]:9.3g} {conv[1]:<4s} {s_repr*conv[0]:10.3g} {conv[1]}")
    summary[lab] = dict(rms_inuse=rms_use, rms_stated=rms_st,
                        sigma_meas_med=float(np.median(s_st))*conv[0],
                        sigma_repr=s_repr*conv[0], unit=conv[1])
out = HERE/"results/error_model_audit.json"
json.dump(dict(map=Path(MAP_PATH).stem, summary=summary), open(out, "w"), indent=1)
print(f"Saved {out}")
