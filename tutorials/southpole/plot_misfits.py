"""plot_misfits.py — South Pole observation-vs-model fit at the frozen r8 MAP.

Runs the engine forward at the r8 MAP and plots, for each assimilated
observable (density, depth-age, layer gradient d(age)/dz, borehole T, ApRES
velocity), the observations with their 1-sigma errors against the model's
kernel predictions (the exact operators the objective uses), plus a combined
standardized-residual panel. This is the paper's fit-quality figure.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/plot_misfits.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)

# build cfg + run forward via run.py's config (exec up to the dispatch).
# FIRN_MAP_JSON selects the MAP (default: frozen r8); FIRN_KNOTS (consumed by
# run.py's cfg builder) must match the MAP's knot layout (e.g. knots_dense.json
# for the dense temporal-probe MAP).
import os
MAP_PATH = os.environ.get("FIRN_MAP_JSON",
                          str(HERE/"results/sp_joint_r8.json"))
MAP_NAME = Path(MAP_PATH).stem
ns = {"__file__": str(HERE/"run.py"), "__name__": "cfgbuild"}
src = open(HERE/"run.py").read().split("warm = json.load")[0]
exec(src, ns)
warm = json.load(open(MAP_PATH))
from firnpack.inverse import assimilate
from firnpack.models.firn import FirnParameters
_p = FirnParameters(); C_I = float(_p.c_i); T_REF = float(_p.T_ref)
YEAR_S = 365.25*24*3600.0
from firnpack.constants import year as YEAR_S  # exact model convention

r = assimilate(ns["cfg"], mode="forward", warm=warm, verbose=False)
print(f"forward J = {r['J']:.3f}; rms: " +
      " ".join(f"{k[4:]}={v:.2f}" for k, v in r["diag"].items() if k.startswith("rms_")))

# unit conversions per block for display
def to_display(block):
    d = np.array(block["depths"]); o = np.array(block["obs"])
    s = np.array(block["sig"]); p = np.array(block["pred"])
    lab = block["label"]
    if lab == "age":      o, s, p = o/YEAR_S, s/YEAR_S, p/YEAR_S; unit = "age (yr)"
    elif lab == "T":      o, s, p = o/C_I + T_REF - 273.15, s/C_I, p/C_I + T_REF - 273.15; unit = "T (°C)"
    elif lab == "rho":    unit = "density (kg m⁻³)"
    elif lab == "dage":   unit = "d(age)/dz (yr m⁻¹)"
    elif lab == "v":      unit = "w (m yr⁻¹)"
    else: unit = lab
    return d, o, s, p, unit

blocks = {b["label"]: b for b in r["obs"]}
order = ["rho", "age", "dage", "T", "v"]
titles = {"rho": "(a) density (SP19)", "age": "(b) depth–age (SP19)",
          "dage": "(c) layer gradient d(age)/dz", "T": "(d) borehole temperature",
          "v": "(e) vertical velocity (ApRES)"}
C_O, C_M, C_R = "#111827", "#2563EB", "#EA580C"

fig, AX = plt.subplots(2, 3, figsize=(15, 9))
for ax, lab in zip(AX.flat[:5], order):
    d, o, s, p, unit = to_display(blocks[lab])
    ax.errorbar(o, d, xerr=s, fmt="o", ms=3.5, color=C_O, elinewidth=0.7,
                capsize=0, alpha=0.75, label="observations ±1σ")
    oo = np.argsort(d)
    ax.plot(p[oo], d[oo], "-", color=C_M, lw=2, label=f"model ({MAP_NAME})")
    ax.invert_yaxis(); ax.set_ylim(132, 0)
    ax.set_xlabel(unit); ax.set_ylabel("depth (m)")
    rms = r["diag"]["rms_"+lab]
    ax.set_title(f"{titles[lab]} — rms {rms:.2f}σ", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (f) all standardized residuals vs depth
ax = AX.flat[5]
cols = {"rho": "#2563EB", "age": "#EA580C", "dage": "#059669", "T": "#7C3AED", "v": "#B45309"}
for lab in order:
    d, o, s, p, _ = to_display(blocks[lab])
    ax.plot((p-o)/s, d, "o", ms=3.5, color=cols[lab], alpha=0.75, label=lab)
ax.axvline(0, color="k", lw=0.8); ax.axvspan(-1, 1, color="gray", alpha=0.12, lw=0)
ax.invert_yaxis(); ax.set_ylim(132, 0); ax.set_xlim(-3.2, 3.2)
ax.set_xlabel("(model − obs) / σ"); ax.set_ylabel("depth (m)")
ax.set_title("(f) standardized residuals, all observables", fontsize=10.5)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

fig.suptitle(f"South Pole joint assimilation — observations vs the {MAP_NAME} MAP "
             f"(J = {r['J']:.1f})", fontsize=13)
fig.tight_layout()
out = FIGS/("sp_misfits.png" if MAP_NAME == "sp_joint_r8"
            else f"sp_misfits_{MAP_NAME}.png")
fig.savefig(out, dpi=140)
print(f"Saved {out}")
