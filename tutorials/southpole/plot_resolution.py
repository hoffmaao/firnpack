"""plot_resolution.py — discretization-sensitivity of the South Pole estimates.

Compares the r8 MAP (NZ=100, dt=5) against re-optimized MAPs at NZ=200 and
dt=2.5 (warm-started from r8). If the recovered stage rates, s2, and the
T/b histories drift far less than the posterior widths, the estimates are
resolution-insensitive. Reads whichever of the result files exist.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)
CASES = [("r8 (NZ=100, dt=5)", HERE.parent.parent/"test/southpole/results/sp_joint_r8.json", "#111827", "-"),
         ("NZ=200, dt=5",      HERE/"results/sp_engine_nz200.json", "#2563EB", "--"),
         ("NZ=100, dt=2.5",    HERE/"results/sp_engine_dt25.json", "#EA580C", ":")]
R, TSITE = 8.314, 222.35  # -50.8 C

def rates(m, b=0.09):
    T = TSITE
    return (m["hl_k0"]*math.exp(-m["hl_Ea1"]/(R*T))*b,
            m["hl_k1"]*math.exp(-m["hl_Ea2"]/(R*T))*math.sqrt(b))

loaded = []
for lab, p, c, ls in CASES:
    if Path(p).exists():
        loaded.append((lab, json.load(open(p)), c, ls))
    else:
        print(f"(missing: {p})")
if len(loaded) < 2:
    raise SystemExit("need at least r8 + one refined case")

base = loaded[0][1]; mb = base["m_map"]; c0b, c1b = rates(mb)
print(f"{'case':22s} {'c0 (stage-1)':>14s} {'c1 (stage-2)':>14s} {'s2':>7s} {'ezz e-5':>8s} {'b_off':>7s}")
rows = []
for lab, r, c, ls in loaded:
    m = r["m_map"]; c0, c1 = rates(m)
    print(f"{lab:22s} {c0:14.4e} {c1:14.4e} {m['s2_shape']:7.3f} {m['ezz_yr']*1e5:8.2f} {m['b_off']:7.3f}")
    rows.append((lab, c0, c1, m))
print("\ndrift vs r8 (%%):")
for lab, c0, c1, m in rows[1:]:
    print(f"  {lab:20s} c0 {100*(c0/c0b-1):+6.2f}%  c1 {100*(c1/c1b-1):+6.2f}%  "
          f"s2 {100*(m['s2_shape']/mb['s2_shape']-1):+6.2f}%  "
          f"kfirn {100*(m['k_firn_scale']/mb['k_firn_scale']-1):+6.2f}%")
print("(pp-round posterior widths for context: rates ~x/1.2, s2 ~x/1.27 — i.e. ~20-27%)")

fig, AX = plt.subplots(1, 3, figsize=(15, 4.8))
ax = AX[0]
labels = ["c0 (stage-1 rate)", "c1 (stage-2 rate)", "s2", "k_firn"]
x = np.arange(len(labels)); w = 0.8/max(len(rows)-1,1)
for i, (lab, c0, c1, m) in enumerate(rows[1:]):
    vals = [c0/c0b-1, c1/c1b-1, m["s2_shape"]/mb["s2_shape"]-1,
            m["k_firn_scale"]/mb["k_firn_scale"]-1]
    ax.bar(x+i*w-0.2, np.array(vals)*100, w, label=lab,
           color=loaded[i+1][2], alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.axhspan(-20, 20, color="gray", alpha=0.10, lw=0)
ax.text(0.02, 0.93, "shaded: ~posterior width (±20%)", transform=ax.transAxes,
        fontsize=8, color="gray")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20)
ax.set_ylabel("drift vs r8 (%)"); ax.set_ylim(-25, 25)
ax.set_title("(a) parameter drift under 2× refinement"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

ax = AX[1]
for lab, r, c, ls in loaded:
    ax.plot(r["knot_years"], r["T_knots"], ls, color=c, lw=1.8, marker="o", ms=3, label=lab)
ax.set_xlabel("year CE"); ax.set_ylabel("surface T (°C)")
ax.set_title("(b) recovered T history"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = AX[2]
for lab, r, c, ls in loaded:
    ax.plot(r["b_knot_years"], r["b_knots"], ls, color=c, lw=1.8, marker="o", ms=3, label=lab)
ax.set_xlabel("year CE"); ax.set_ylabel("accumulation (m ice/yr)")
ax.set_title("(c) recovered b(t)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("South Pole estimates vs discretization — MAPs re-optimized at refined NZ / dt", fontsize=12)
fig.tight_layout()
out = FIGS/"resolution_sensitivity.png"
fig.savefig(out, dpi=140); print(f"Saved {out}")
