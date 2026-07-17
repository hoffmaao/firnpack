"""diag_dage_sigma_origin.py — where does sigma_dage come from, and what SHOULD it be?

MODEL-FREE. Uses only sp19_depth_age.csv (already in use) and no external
uncertainty column, no Desroziers, no MAP. The question is internal:

HISTORICAL: this is the diagnostic that FOUND the sigma_dage fix, so it rebuilds
the PRE-FIX sigma (the window s.e. floored at 4% of the slope) and measures what
it missed. run.py no longer builds sigma that way -- it now uses
sqrt(sigma_meas^2 + sigma_repr^2), which is this script's conclusion applied
(FIRN_SIG_DAGE_LEGACY=1 restores the old form). Read this as the record of the
argument, not as a description of the current error model.

The pre-fix sigma_dage was the standard error of a straight-line fit through
the ~12 annual layers inside a +-0.6 m window (floored at 4% of the slope).
That is NOT a measurement error -- it is an estimate of interannual layer
variability, but ONLY at the sub-1.2 m scale.

The model, however, cannot track layers across the WHOLE decadal-and-below
band (smooth b-knots; the d(age)/dz operator smears sub-decadal b). So the
representativeness error it faces is the variance of d(age)/dz about the
smoothest curve the model can actually produce -- a much wider band than the
window s.e. samples.

Measure the variance of d(age)/dz as a function of averaging scale (a structure
function). If the window s.e. samples only the shortest scales, it will sit
BELOW the variance in the band the model must absorb, by a factor we can read
off directly -- an internal explanation for the misfit, owing nothing to an
external source.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diag_dage_sigma_origin.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not diagnostics/
H0 = 130.0

a = pd.read_csv(HERE / "data" / "sp19_depth_age.csv")
a["age_yr"] = 2015.0 - a.year_CE
a = a.query("depth_m<=@H0 and age_yr>=0").sort_values("depth_m").reset_index(drop=True)
ad, aa = a.depth_m.values, a.age_yr.values

# ---- 1. rebuild the observable, and the PRE-FIX (legacy) sigma -------------
# The observable is still run.py's; the sigma is the old floored form this
# script exists to indict, NOT the sqrt(dse^2 + repr^2) run.py now uses.
cent = np.arange(6.0, H0 - 1.0 + 1e-9, 1.0)
obs, sig, nwin = [], [], []
for c in cent:
    s = np.abs(ad - c) <= 0.6
    if s.sum() >= 4:
        A = np.vstack([ad[s] - c, np.ones(s.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, aa[s], rcond=None)
        nn = s.sum()
        se = math.sqrt(max(float(res[0]) if len(res) else 0.0, 1e-12) / (nn - 2)
                       / max(np.sum((ad[s] - c) ** 2), 1e-12))
        obs.append(coef[0]); sig.append(max(se, 0.04 * abs(coef[0]))); nwin.append(nn)
z = cent[:len(obs)]; obs = np.array(obs); sig = np.array(sig); nwin = np.array(nwin)

print(f"rebuilt the dage block with the LEGACY sigma: {len(z)} pts, "
      f"{z.min():.0f}-{z.max():.0f} m")
print(f"  layers per +-0.6 m window : median {np.median(nwin):.0f}")
print(f"  d(age)/dz                 : median {np.median(obs):.2f} yr/m")
print(f"  legacy sigma (window s.e.): median {np.median(sig):.3f} yr/m "
      f"({100*np.median(sig/obs):.1f}% of the value)")
print(f"  fraction of points on the 4% FLOOR (not the s.e.): "
      f"{100*np.mean(sig <= 0.04*np.abs(obs)+1e-12):.0f}%")

# ---- 2. structure function: variance of dage vs averaging scale ------------
# smooth the OBSERVED dage with running means of increasing length; the
# variance removed at each scale is the power living below that scale.
print(f"\n===== variance of d(age)/dz by scale (model-free) =====")
print(f"{'scale':>8} {'~years':>8} {'sd about the smooth (yr/m)':>28}")
rows = []
for L in (2, 4, 7, 10, 15, 20, 30, 50):
    w = max(3, int(round(L / 1.0)) | 1)
    if w >= len(obs): continue
    sm = np.convolve(np.pad(obs, w // 2, mode="edge"), np.ones(w) / w, mode="valid")[:len(obs)]
    sd = float(np.std(obs - sm))
    yrs = L * float(np.median(obs))          # metres -> years via dage
    rows.append((L, yrs, sd))
    print(f"{L:6d} m {yrs:8.0f} {sd:28.3f}")

# ---- 3. the comparison that matters ---------------------------------------
sig_med = float(np.median(sig))
print(f"\n===== the origin of the 2.2x =====")
print(f"the legacy sigma samples ONLY the +-0.6 m window: {sig_med:.3f} yr/m")
for L, yrs, sd in rows:
    print(f"  variability below {L:2d} m (~{yrs:4.0f} yr): {sd:.3f} yr/m "
          f"= {sd/sig_med:5.2f}x the legacy sigma")

# the model's b-knots are ~68 yr apart (15 knots over 1015 yr) -> in metres:
knot_m = 68.0 / float(np.median(obs))
print(f"\nthe b-knots resolve ~68 yr = ~{knot_m:.1f} m of core, so EVERYTHING")
print(f"below ~{knot_m:.1f} m is unfittable and must live in sigma.")
Lb = min(rows, key=lambda r: abs(r[0] - knot_m))
print(f"  variability below ~{Lb[0]} m = {Lb[2]:.3f} yr/m "
      f"-> sigma should be ~{Lb[2]/sig_med:.2f}x the legacy one (run.py now does this)")
print("\nThis is an INTERNAL account of the misfit: the sigma and the model's")
print("representativeness gap are measured at different scales. It uses no")
print("external uncertainty column and no Desroziers step.")

json.dump(dict(sigma_inuse_med=sig_med, dage_med=float(np.median(obs)),
               knot_scale_m=knot_m,
               structure=[dict(scale_m=L, years=y, sd=s) for L, y, s in rows]),
          open(HERE / "output" / "dage_sigma_origin.json", "w"), indent=1)
print(f"\nSaved {HERE/'output'/'dage_sigma_origin.json'}")
