"""Why do the thermal (firn-T-controlling) parameters drift from truth?

The borehole-T profile is fit jointly by the conductivity scales (k_snow,
k_firn), the basal flux (Q_base), the T-dependent stage-2 densification
(hl_Ea2), and the surface-T history. If several of these produce the SAME
borehole-T response, they are degenerate — interchangeable in the fit — so the
inversion can trade them freely and they wander off truth even when the
temperature RECONSTRUCTION (a different combination) is fine.

Method: at the recovered MAP, perturb each parameter and record the borehole-T
response VECTOR (at the obs depths). Report
  - response magnitude in sigma (is it constrained at all?),
  - pairwise cosine similarity of the response vectors (are they collinear?).
cos ~ 1 between two params => they trade => joint degeneracy.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/synthetic/diagnostics/thermal_degeneracy.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from firnpack import plot as fp
from firnpack.inverse import SiteConfig, ObsBlock, ScalarCtrl, KnotCtrl, assimilate

OUT = HERE / "output"
# newest OSSE MAP by mtime, same rule as plot.py / reanalysis.py; env override
_mapp = fp.newest_map(OUT, need=("m_map",), env="FIRN_OSSE_MAP",
                      pattern="synthetic_osse*.json")
if _mapp is None:
    raise SystemExit(f"no OSSE MAP in {OUT} — run tutorials/synthetic/run.py first")
print(f"thermal-degeneracy MAP: {_mapp.name}")
mp = json.load(open(_mapp))
m = mp["m_map"]
Ty = np.array(mp["knot_years"], float); Tv = np.array(mp["T_knots"], float)
By = np.array(mp["b_knot_years"], float); Bv = np.array(mp["b_knots"], float)
SCAL = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_snow_scale", "k_firn_scale", "Q_base", "s2_shape"]
EZZ = float(m.get("ezz_yr", 0.0))
zobs = np.arange(13., 129., 7.)
SIG = np.interp(zobs, [13, 30, 45, 60, 80, 100, 115, 128],
                [.027, .027, .050, .068, .060, .045, .035, .055])


def bore(scal_over, Tvals):
    scal = [ScalarCtrl(k, scal_over.get(k, m[k]), scal_over.get(k, m[k]), 1.0, 1e-6, 1e9,
                       log=(k != "Q_base"), active=False) for k in SCAL]
    if EZZ:
        scal.append(ScalarCtrl("ezz_yr", EZZ, EZZ, 1.0, -3e-4, 2e-4, active=False))
    cfg = SiteConfig(name="deg", out_dir=str(OUT), tag="degtest", H_col=130.0, NZ=100,
                     spin_years=2500.0, dt_years=5.0, rho_surf=350.0, rho_ic_deep=820.0,
                     rho_ic_scale=25.0, conductivity_law="calonne2019", scalars=scal,
                     T_knots=KnotCtrl(Ty, Tvals, Tvals, 0.6, -60, -40, invert=False, name="Tk"),
                     b_knots=KnotCtrl(By, Bv, Bv, 0.2, 0.03, 0.20, log=True, invert=False, name="b"),
                     obs=[ObsBlock("rho", np.array([50.]), np.array([600.]), np.array([1e3]), label="rho")])
    r = assimilate(cfg, mode="forward", verbose=False)
    p = r["profiles"]; d = np.array(p["depth"]); o = np.argsort(d)
    return np.interp(zobs, d[o], np.array(p["T_C"])[o])


base = bore({}, Tv)
# perturb each thermal/law param (+5%) and a recent + old T-knot (+0.5 K)
probes = {"k_snow": ("k_snow_scale", m["k_snow_scale"]*0.05),
          "k_firn": ("k_firn_scale", m["k_firn_scale"]*0.05),
          "Q_base": ("Q_base", 0.005),
          "hl_Ea2": ("hl_Ea2", m["hl_Ea2"]*0.05),
          "T_2010": ("Tk", 0.5), "T_1000": ("Tk0", 0.5)}
resp = {}
for name, (key, dv) in probes.items():
    if key == "Tk":
        Tp = Tv.copy(); Tp[list(Ty).index(2010.)] += dv; r = bore({}, Tp)
    elif key == "Tk0":
        Tp = Tv.copy(); Tp[0] += dv; r = bore({}, Tp)
    else:
        r = bore({key: m[key] + dv}, Tv)
    resp[name] = (r - base) / SIG                      # response in sigma units
    print(f"  {name:8s} response magnitude = {np.sqrt(np.mean(resp[name]**2)):.2f} sigma")

names = list(resp)
print("\npairwise cosine similarity of borehole-T response vectors "
      "(|cos|~1 => degenerate / interchangeable):")
print("        " + "".join(f"{n:>8s}" for n in names))
C = np.zeros((len(names), len(names)))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        va, vb = resp[a], resp[b]
        C[i, j] = np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb)+1e-30)
    print(f"{a:8s}" + "".join(f"{C[i,j]:+8.2f}" for j in range(len(names))))
json.dump(dict(names=names, cos=C.tolist(),
               resp_sigma={k: float(np.sqrt(np.mean(v**2))) for k, v in resp.items()}),
          open(OUT/"thermal_degeneracy.json", "w"), indent=1)
print(f"\nSaved {OUT/'thermal_degeneracy.json'}")
