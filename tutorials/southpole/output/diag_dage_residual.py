"""diag_dage_residual.py — the STRUCTURE of the d(age)/dz misfit, panel (c).

Panel (c) is the biggest misfit on the figure (2.2 sigma) and d(age)/dz is the
observable that constrains b(t), so what KIND of 2.2 sigma it is decides
whether b(t) is trustworthy.

Split the residual by SCALE rather than by site:

    r(z) = smooth(z) + wiggle(z)          (smooth = moving average over LEN_M)

  smooth(z) = depth-coherent part. The model CAN represent this (b-knots are
              multidecadal; a layer is ~10 yr/m, so 10 m ~ a century). If this
              dominates -> a real model/forcing bias, and b(t) is biased.
  wiggle(z) = sub-decadal oscillation. The model CANNOT represent it at any
              control setting (memory: d(age)/dz operator has a broad NULL
              SPACE at sub-decadal scales; b spikes are smeared ~4x). If this
              dominates -> representativeness, not error: the honest fix is
              the error model, not the physics.

Also: is the residual WHITE at the 1-m sampling (lag-1 autocorrelation)?

HISTORICAL: this diagnostic was run against the PRE-FIX error model, whose
sigma was a window-fit s.e. + 4% floor with NO representativeness term -- the
leading hypothesis for the 2.2x, and the one it confirmed. run.py now carries
that term (sqrt(sigma_meas^2 + sigma_repr^2)), so re-running this against a MAP
produced under the new sigma should show rms near 1 and the inflation factor
below near 1.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diag_dage_residual.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not output/
import sys
sys.path.insert(0, str(HERE))  # sibling config module
from config import build_cfg
R = HERE / "results"
MAP_PATH = os.environ.get("FIRN_WARM_JSON", str(R / "sp_r10_final.json"))
LEN_M = float(os.environ.get("FIRN_SMOOTH_M", "10.0"))   # coherence length

os.environ.update(FIRN_VEL_SITE="x17s2+x11n0+x11n2+x11n6+x11s2",
                  FIRN_VEL_SRC="zeising", FIRN_SEAS="1", FIRN_HCOL="300",
                  FIRN_EZZ_SITE="none")   # r10 ran with ezz free; score it as it ran
_b = build_cfg()

from firnpack.inverse import assimilate
warm = json.load(open(MAP_PATH))
r = assimilate(_b.cfg, mode="forward", warm=warm, verbose=False)
b = {x["label"]: x for x in r["obs"]}["dage"]
z, o, p, s = (np.array(b[k]) for k in ("depths", "obs", "pred", "sig"))
res = (p - o) / s
print(f"\nd(age)/dz at {Path(MAP_PATH).name}: N={len(z)}, "
      f"{z.min():.0f}-{z.max():.0f} m, rms {np.sqrt(np.mean(res**2)):.2f} sigma")

# ---- 1. scale split ---------------------------------------------------------
dz = float(np.median(np.diff(z)))
w = max(3, int(round(LEN_M / dz)) | 1)                 # odd window
kern = np.ones(w) / w
pad = np.pad(res, w // 2, mode="edge")
smooth = np.convolve(pad, kern, mode="valid")
wiggle = res - smooth
v_s, v_w = float(np.mean(smooth**2)), float(np.mean(wiggle**2))
print(f"\n===== scale decomposition (smoothing length {LEN_M:.0f} m, "
      f"{w} pts) =====")
print(f"  smooth (model CAN fit)  <smooth^2> = {v_s:6.3f}  "
      f"({100*v_s/(v_s+v_w):4.1f}%)   rms {np.sqrt(v_s):.2f} sigma")
print(f"  wiggle (model CANNOT)   <wiggle^2> = {v_w:6.3f}  "
      f"({100*v_w/(v_s+v_w):4.1f}%)   rms {np.sqrt(v_w):.2f} sigma")
print(f"  -> {'MODEL/FORCING BIAS: b(t) is suspect' if v_s > v_w else 'REPRESENTATIVENESS: unfittable layering, fix the error model'}")

# ---- 2. whiteness -----------------------------------------------------------
def acf(x, k):
    x = x - x.mean()
    return float(np.sum(x[:-k] * x[k:]) / np.sum(x**2))
print(f"\n===== whiteness of the raw residual (1-m sampling) =====")
for k in (1, 2, 3, 5):
    print(f"  lag-{k} autocorr = {acf(res, k):+.3f}")
print("  (white -> the stated sigma is just too small by the rms factor;")
print("   strongly correlated -> a coherent structure the model is missing)")

# ---- 3. where does the smooth part live? ------------------------------------
print(f"\n===== smooth (coherent) residual vs depth =====")
print(f"{'z band':>14} {'N':>4} {'mean res':>9} {'rms res':>8}")
for lo in range(0, 130, 20):
    m = (z >= lo) & (z < lo + 20)
    if m.sum():
        print(f"{f'{lo}-{lo+20} m':>14} {m.sum():4d} {res[m].mean():+9.2f} "
              f"{np.sqrt(np.mean(res[m]**2)):8.2f}")
print(f"\n  overall mean residual (bias) = {res.mean():+.3f} sigma")

# ---- 4. what sigma would make this fit? -------------------------------------
infl = float(np.sqrt(np.mean(res**2)))
print(f"\n===== error-model implication =====")
print(f"  Desroziers-style inflation to reach rms 1: sigma x {infl:.2f}")
print(f"  (the archived r8 used x2.2 by hand; Andrew 07-10 dropped it to show")
print(f"   the honest misfit. This says the STATED sigma omits a")
print(f"   representativeness term of ~{np.sqrt(max(infl**2-1,0)):.2f}x the stated value.)")

out = R / "dage_residual_structure.json"
json.dump(dict(map=Path(MAP_PATH).name, n=len(z), rms=infl,
               var_smooth=v_s, var_wiggle=v_w, smooth_len_m=LEN_M,
               acf1=acf(res, 1), bias=float(res.mean()),
               depths=z.tolist(), res=res.tolist()), open(out, "w"), indent=1)
print(f"\nSaved {out}")
