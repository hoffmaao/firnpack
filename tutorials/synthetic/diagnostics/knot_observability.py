"""knot_observability.py — WHY is the initial part of the T record not recovered?

Direct test of the thermal null space: perturb each surface-T knot by +1 K in
the truth forward and measure the RMS change it produces in the BOREHOLE-T
observations (13-125 m). A knot the borehole cannot see (response ~ 0) is
unrecoverable at ANY signal size — its recovered value must revert to the prior.

Reports, per knot year:
  - response (mK per K) at the borehole depths,
  - the observation window it maps to (depth of peak response),
so the drop-off with age quantifies exactly how far back the firn "remembers".

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/synthetic/diagnostics/knot_observability.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from firnpack.inverse.osse import truth_profiles

OUT = HERE / "output"
tr = json.load(open(OUT / "synthetic_truth.json"))
Ty = np.array(tr["T_years"], float); Tv = np.array(tr["T_truth"], float)
By = np.array(tr["b_years"], float); Bv = np.array(tr["b_truth"], float)
ez = float(tr.get("ezz_truth", 0.0))

Td_d = np.arange(13.0, 129.0, 7.0)   # the borehole obs depths
# borehole sigma (for response in sigma units, not just mK)
Td_sig = np.interp(Td_d, [13., 30., 45., 60., 80., 100., 115., 128.],
                   [0.027, 0.027, 0.050, 0.068, 0.060, 0.045, 0.035, 0.055])


def bore_T(Tvals):
    d, _, _, T = truth_profiles(str(OUT), T_years=Ty, T_vals=Tvals,
                                B_years=By, B_vals=Bv, ezz_truth=ez)
    return np.interp(Td_d, d, T)


base = bore_T(Tv)
print(f"knot observability (borehole T response to +1 K per knot; "
      f"sigma {1000*Td_sig.min():.0f}-{1000*Td_sig.max():.0f} mK):")
print(f"{'year':>6s}{'response_mK/K':>15s}{'in_sigma':>10s}{'peak_depth_m':>14s}  observable?")
rows = []
for i, yr in enumerate(Ty):
    Tp = Tv.copy(); Tp[i] += 1.0
    dT = bore_T(Tp) - base
    rms_mK = float(np.sqrt(np.mean(dT ** 2)) * 1000)
    in_sig = float(np.sqrt(np.mean((dT / Td_sig) ** 2)))   # response in sigma units
    zpk = float(Td_d[int(np.argmax(np.abs(dT)))])
    tag = "YES" if in_sig > 1.0 else ("marginal" if in_sig > 0.3 else "NO — null space")
    rows.append((yr, rms_mK, in_sig, zpk, tag))
    print(f"{yr:6.0f}{rms_mK:15.1f}{in_sig:10.2f}{zpk:14.0f}  {tag}")

json.dump([dict(year=r[0], resp_mK=r[1], resp_sigma=r[2], peak_depth=r[3]) for r in rows],
          open(OUT / "knot_observability.json", "w"), indent=1)
obs_cut = max((r[0] for r in rows if r[2] > 1.0), default=None)
print(f"\nThe borehole constrains surface T back to ~{obs_cut:.0f} CE "
      f"(response > 1 sigma); everything older is the NULL SPACE -> prior-bound.")
print(f"Saved {OUT/'knot_observability.json'}")
