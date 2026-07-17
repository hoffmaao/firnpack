"""tutorials/southpole/run.py — South Pole assimilation via the shared engine.

South Pole joint assimilation through firnpack.inverse.assimilate: coarse knots
(knots_r8), corrected borehole-T datum (-5.8), k_snow pinned (USP50 1.29),
de-biased accumulation prior (Buizert climatology 0.096), Calonne conductivity,
per-point chi^2.

The DEFAULT config is the one we believe: data-derived sigma_dage, no
absolute-age block, the x11n6 differenced Zeising velocity block, and ezz pinned
to x11n6's strain measured below close-off. It is NOT the archived-MAP config.

The frozen r8 MAP (results/sp_joint_r8.json) is still reproducible exactly, at
J = 81.3061, under the legacy guards:

    FIRN_SIG_DAGE_LEGACY=1 FIRN_SIG_DAGE_SCALE=2.2 FIRN_AGE_BLOCK=1 \\
    FIRN_BASAL=Q FIRN_EZZ_SITE=none FIRN_VEL_SITE=pooled

Validate mode checks against that J only when all of those are set; otherwise
there is no target to check, because the error model has changed by design.

Modes (env FIRN_MODE): "validate" (default; forward J at the warm MAP),
"verify" (replay + FD), "optimize" (full run, warm from r8).
Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/run.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
from firnpack.inverse import assimilate

sys.path.insert(0, str(Path(__file__).parent))  # for the sibling config module
from config import build_cfg

_b = build_cfg()
cfg, MODE, NZ, DT = _b.cfg, _b.MODE, _b.NZ, _b.DT
SP_RESULTS, EZZ_ZMIN, EZZ_ZMAX = _b.SP_RESULTS, _b.EZZ_ZMIN, _b.EZZ_ZMAX
VEL_SITE, EZZ_SITE = _b.VEL_SITE, _b.EZZ_SITE
SIG_DAGE_SCALE, _LEGACY_DAGE = _b.SIG_DAGE_SCALE, _b._LEGACY_DAGE

warm = json.load(open(os.environ.get("FIRN_WARM_JSON", str(SP_RESULTS/"sp_joint_r8.json"))))

# r8 is an ARCHIVED MAP under an ARCHIVED error model: its J is a meaningful
# target only when every legacy guard that defined that model is set. Under the
# defaults the error model has changed by design, so there is nothing to match.
_R8_REPRO = (_LEGACY_DAGE and abs(SIG_DAGE_SCALE - 2.2) < 1e-9
             and os.environ.get("FIRN_AGE_BLOCK","0")=="1"
             and os.environ.get("FIRN_BASAL","G")=="Q"
             and EZZ_SITE=="none" and VEL_SITE=="pooled")

if MODE=="validate":
    r=assimilate(cfg, mode="forward", warm=warm)
    print(f"\nENGINE forward J at the warm MAP = {r['J']:.4f}  (NZ={NZ} dt={DT})")
    print(f"  rms: " + " ".join(f"{k[4:]}={v:.3f}" for k,v in r['diag'].items() if k.startswith('rms_')))
    p=r["profiles"]; d=np.array(p["depth"]); rho=np.array(p["rho"]); ag=np.array(p["age_yr"]); Tp=np.array(p["T_C"])
    zco=float(np.interp(830.0,rho,d)) if rho.max()>830 else float("nan")
    print(f"  derived: close-off(830)={zco:.2f} m  age(100m)={np.interp(100,d,ag):.1f} yr  "
          f"rho(50m)={np.interp(50,d,rho):.1f}  T(bot)={Tp[-1]:.3f} C")
    if _R8_REPRO:
        print(f"  match to r8 (target J = 81.3061): "
              f"{'YES' if abs(r['J']-81.3061)<0.5 else 'CHECK'}")
    else:
        print("  no r8 target: the default error model is not r8's. To reproduce r8, set")
        print("    FIRN_SIG_DAGE_LEGACY=1 FIRN_SIG_DAGE_SCALE=2.2 FIRN_AGE_BLOCK=1 "
              "FIRN_BASAL=Q FIRN_EZZ_SITE=none FIRN_VEL_SITE=pooled")
elif MODE=="verify":
    _basal = "Q_base" if os.environ.get("FIRN_BASAL","G")=="Q" else "G_base"
    _fdn = os.environ.get("FIRN_FD_NAMES")
    assimilate(cfg, mode="verify", warm=warm,
               fd_names=(_fdn.split(",") if _fdn else
                         [_basal,"k_firn_scale","s2_shape","b1960","Tk8"]
                         + (["k_snow_scale"] if os.environ.get("FIRN_SEAS","0")=="1" else [])),
               fd_h=float(os.environ.get("FIRN_FD_H", "1e-3")))
elif MODE=="optimize":
    assimilate(cfg, mode="optimize", warm=warm)
