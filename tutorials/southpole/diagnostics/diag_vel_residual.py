"""diag_vel_residual.py - the STRUCTURE of the velocity misfit, panel (e).

Not the rms: the shape. At the r10 MAP (velocity ON tape, so the model has
already done its best), decompose the per-site residual r_s(z) into

    r_s(z) = common(z) + site_s(z)

  common(z) = mean over sites      -> if this dominates, the residual is OURS
                                      (model w(z) curvature or the dRdt_diff
                                      operator); no site choice can fix it.
  site_s(z) = deviation from it    -> if this dominates, the residual is the
                                      array's real spatial variation, and
                                      site SELECTION is the lever.

The observable is differenced about z_ref, so r(z_ref) == 0 by construction:
structure must be read as growth AWAY from the reference depth, not as offset.

Env: FIRN_WARM_JSON (default output/sp_r10_final.json), FIRN_DIAG_SITES.
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/diagnostics/diag_vel_residual.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not diagnostics/
import sys
sys.path.insert(0, str(HERE))  # sibling config module
from config import build_cfg
R = HERE / "output"
# r10_final's 5 blocks, in the order its log built them
SITES = os.environ.get("FIRN_DIAG_SITES", "x17s2+x11n0+x11n2+x11n6+x11s2")
MAP_PATH = os.environ.get("FIRN_WARM_JSON", str(R / "sp_r10_final.json"))

os.environ.update(FIRN_VEL_SITE=SITES, FIRN_VEL_SRC="zeising",
                  FIRN_SEAS="1", FIRN_HCOL="300",   # H300 verified for r10
                  FIRN_EZZ_SITE="none")             # r10 ran with ezz free
_b = build_cfg()

from firnpack.inverse import assimilate
warm = json.load(open(MAP_PATH))
r = assimilate(_b.cfg, mode="forward", warm=warm, verbose=False)
print(f"\nforward at {Path(MAP_PATH).name}: J={r['J']:.2f}  " +
      " ".join(f"{k[4:]}={v:.2f}" for k, v in r["diag"].items()
               if k.startswith("rms_")))

vb = [b for b in r["obs"] if b["label"].startswith("v_")]
# ref_depth lives on the cfg's ObsBlock, not in the forward's obs dict
zref = float([o for o in _b.cfg.obs
              if o.label == vb[0]["label"]][0].ref_depth)

# residuals on a common depth axis (all sites share the zeising range grid)
Z = np.array(vb[0]["depths"])
RES, MM = {}, {}
for b in vb:
    s = b["label"][2:]
    o, p, sg = (np.array(b[k]) for k in ("obs", "pred", "sig"))
    RES[s] = (p - o) / sg
    MM[s] = (p - o) * 1000.0        # mm/yr
names = list(RES)
Rm = np.array([RES[s] for s in names])          # (nsite, ndepth), sigma units
Mm = np.array([MM[s] for s in names])           # mm/yr

common = Rm.mean(0)
sitedev = Rm - common
print(f"\nref depth = {zref:.1f} m  (residual is 0 there BY CONSTRUCTION)")
print(f"\n{'z(m)':>7} " + " ".join(f"{s:>8}" for s in names)
      + f" | {'common':>8} {'site sd':>8}")
for j in range(len(Z)):
    print(f"{Z[j]:7.1f} " + " ".join(f"{Rm[i, j]:8.2f}" for i in range(len(names)))
          + f" | {common[j]:8.2f} {sitedev[:, j].std():8.2f}")

# ---- the decomposition: which term carries the variance? -------------------
v_common = float(np.mean(common**2))
v_site = float(np.mean(sitedev**2))
print("\n===== variance decomposition (sigma^2 units) =====")
print(f"  common-mode  <common^2>  = {v_common:6.3f}   "
      f"({100*v_common/(v_common+v_site):4.1f}%)")
print(f"  site-specific <dev^2>    = {v_site:6.3f}   "
      f"({100*v_site/(v_common+v_site):4.1f}%)")
print(f"  -> the residual is {'MOSTLY OURS (model/operator)' if v_common > v_site else 'MOSTLY THE ARRAY (site variation)'}")

_sig_med = float(np.median(np.concatenate([np.array(b["sig"]) for b in vb])))*1000.0
print(f"\ncommon-mode in physical units: rms {np.sqrt(np.mean((Mm.mean(0))**2)):.2f} mm/yr"
      f"  (vs the in-use sigma, median {_sig_med:.2f} mm/yr across these blocks -"
      f" per-site now, not a single assumed 3.5)")
print(f"peak |common| = {np.abs(Mm.mean(0)).max():.2f} mm/yr at z = "
      f"{Z[np.argmax(np.abs(Mm.mean(0)))]:.1f} m")

# ---- is the common mode a coherent SHAPE (not noise)? ----------------------
# fit common(z) in mm/yr with a linear + quadratic in (z - zref)
x = Z - zref
c_mm = Mm.mean(0)
for deg, lbl in [(1, "linear"), (2, "quadratic")]:
    co = np.polyfit(x, c_mm, deg)
    resid = c_mm - np.polyval(co, x)
    ss = 1 - np.sum(resid**2)/np.sum((c_mm - c_mm.mean())**2)
    print(f"  {lbl:>9} fit of common(z) in (z-zref): R^2 = {ss:5.3f}")
print("\n  A high quadratic R^2 = a smooth curvature error -> the model's w(z)")
print("  shape or the operator, NOT scatter. That is a physics lead.")

out = R / "vel_residual_structure.json"
json.dump(dict(map=Path(MAP_PATH).name, ref_depth=zref, sites=names,
               depths=Z.tolist(), common_sigma=common.tolist(),
               common_mm_yr=c_mm.tolist(),
               var_common=v_common, var_site=v_site), open(out, "w"), indent=1)
print(f"\nSaved {out}")
