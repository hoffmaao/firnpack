"""Synthetic OSSE figure, rebuilt from output/.

One of the tutorial's two scripts: run.py recovers a known truth from noisy
synthetic data and writes output/, this draws the recovery figure. It reads
JSON only -- no Firedrake, no solve.

The figure is the method-credibility anchor: densification parameters,
surface-T history and accumulation history, truth vs recovered. It needs only
the recovered m_map and knots, so it does not depend on the engine's stored
obs/pred blocks.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 python tutorials/synthetic/plot.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

from firnpack import plot as fp

HERE = Path(__file__).parent
R = HERE / "output"
FIGS = HERE / "figures"; FIGS.mkdir(exist_ok=True)

MAP_PATH = R / "synthetic_osse.json"
TRUTH_PATH = R / "synthetic_truth.json"
if not (MAP_PATH.exists() and TRUTH_PATH.exists()):
    missing = [str(p) for p in (MAP_PATH, TRUTH_PATH) if not p.exists()]
    raise SystemExit(f"nothing to plot — run run.py first (missing {', '.join(missing)})")

mp = json.load(open(MAP_PATH))
tr = json.load(open(TRUTH_PATH))
m = mp["m_map"]; TRUTH = tr["truth"]
C_T, C_R = fp.OBS, fp.MODEL  # truth in ink, recovered in model-blue

fig, AX = plt.subplots(1, 3, figsize=(15, 4.8))

# (a) scalar params: recovered/truth ratio (log axis for the rate params)
ax = AX[0]
keys = [k for k in TRUTH if k in m]
xr = np.arange(len(keys)); ratio = [m[k]/TRUTH[k] for k in keys]
ax.axhline(1.0, color=C_T, lw=1)
ax.bar(xr, ratio, color=C_R, alpha=0.8, width=0.6)
for i, k in enumerate(keys):
    ax.text(i, ratio[i]+0.02, f"{ratio[i]:.2f}", ha="center", fontsize=8)
ax.set_xticks(xr); ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("recovered / truth"); ax.set_ylim(0.7, 1.3)
ax.set_title("(a) densification + conductivity params"); ax.grid(alpha=0.3, axis="y")

# (b) T-history
ax = AX[1]
ty = np.array(tr["T_years"]); tv = np.array(tr["T_truth"])
ry = np.array(mp["knot_years"]); rv = np.array(mp["T_knots"])
ax.plot(ty, tv, "o-", color=C_T, lw=2, ms=5, label="truth")
ax.plot(ry, rv, "s--", color=C_R, lw=1.8, ms=5, label="recovered")
ax.set_xlabel("year CE"); ax.set_ylabel("surface T (°C)")
ax.set_title("(b) surface-temperature history"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (c) accumulation
ax = AX[2]
by = np.array(tr["b_years"]); bv = np.array(tr["b_truth"])
rby = np.array(mp["b_knot_years"]); rbv = np.array(mp["b_knots"])
ax.plot(by, bv, "o-", color=C_T, lw=2, ms=5, label="truth")
ax.plot(rby, rbv, "s--", color=C_R, lw=1.8, ms=5, label="recovered")
ax.set_xlabel("year CE"); ax.set_ylabel("accumulation (m ice/yr)")
ax.set_title("(c) accumulation history"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.suptitle("Synthetic OSSE — the engine recovers known truth (densification law, "
             "T-history, accumulation) from noisy synthetic data", fontsize=12)
fig.tight_layout(); fig.savefig(FIGS/"osse_recovery.png", dpi=140)
print("recovery (recovered/truth):")
for k in keys: print(f"  {k:12s} {m[k]/TRUTH[k]:.3f}")
print(f"Saved {FIGS/'osse_recovery.png'}")
