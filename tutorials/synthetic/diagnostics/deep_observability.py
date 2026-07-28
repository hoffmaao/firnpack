"""Does deepening the domain + deep-ice borehole T make the OLD surface T observable?

Forward-only test: run the truth at H_col = 130 m (firn only) and 300 m (into
the deep ice), and measure the T response to each surface-T knot at (a) the
shallow firn obs depths (13-125 m) and (b) shallow + deep-ice obs (13-290 m).
If the old-knot response rises above 1 sigma only when deep obs are added, the
deep ice carries recoverable old-surface-T information.
"""
import json, sys
from pathlib import Path
import numpy as np
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate

OUT = "/home/andrew/projects/firnpack/tutorials/synthetic/output"
tr = json.load(open(OUT + "/synthetic_truth.json"))
TRUTH = tr["truth"]
Ty = np.array(tr["T_years"], float); Tv = np.array(tr["T_truth"], float)
By = np.array(tr["b_years"], float); Bv = np.array(tr["b_truth"], float)
EZZ = float(tr.get("ezz_truth", 0.0))
SIG = 0.045   # representative borehole sigma (K) for the response-in-sigma metric


def profiles(Tvals, H_col, NZ):
    scal = [ScalarCtrl(k, TRUTH[k], TRUTH[k], 1.0, 1e-6, 1e9,
                       log=(k != "Q_base"), active=False) for k in TRUTH]
    if EZZ:
        scal.append(ScalarCtrl("ezz_yr", EZZ, EZZ, 1.0, -3e-4, 2e-4, active=False))
    cfg = SiteConfig(name="deep", out_dir=OUT, tag="deeptest", H_col=H_col, NZ=NZ,
                     spin_years=2500.0, dt_years=5.0, rho_surf=350.0, rho_ic_deep=820.0,
                     rho_ic_scale=25.0, conductivity_law="calonne2019", scalars=scal,
                     T_knots=KnotCtrl(Ty, Tvals, Tvals, 0.6, -60, -40, invert=False, name="Tk"),
                     b_knots=KnotCtrl(By, Bv, Bv, 0.2, 0.03, 0.20, log=True, invert=False, name="b"),
                     obs=[ObsBlock("rho", np.array([50.0]), np.array([600.]), np.array([1e3]), label="rho")])
    r = assimilate(cfg, mode="forward", verbose=False)
    p = r["profiles"]; d = np.array(p["depth"]); o = np.argsort(d)
    return d[o], np.array(p["age_yr"])[o], np.array(p["T_C"])[o]


for H, NZ, zobs, tag in [(130.0, 100, np.arange(13, 129, 7.), "firn-only (13-125 m)"),
                         (300.0, 160, np.arange(13, 291, 10.), "deep (13-290 m)")]:
    d0, age0, T0 = profiles(Tv, H, NZ)
    print(f"\n=== {tag}: H_col={H:.0f} m, age at base = {age0[np.argmax(d0)]:.0f} yr "
          f"(deposited ~{2015-age0[np.argmax(d0)]:.0f} CE) ===")
    base = np.interp(zobs, d0, T0)
    print(f"{'knot yr':>8s}{'resp (sigma)':>14s}   observable?")
    for i, yr in enumerate(Ty):
        if yr not in (1000., 1300., 1600., 1850., 1950., 2010.):
            continue
        Tp = Tv.copy(); Tp[i] += 1.0
        d1, _, T1 = profiles(Tp, H, NZ)
        resp = float(np.sqrt(np.mean(((np.interp(zobs, d1, T1) - base) / SIG) ** 2)))
        tagr = "YES" if resp > 1 else ("marginal" if resp > 0.3 else "NO")
        print(f"{yr:8.0f}{resp:14.2f}   {tagr}")
