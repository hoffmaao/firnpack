"""plot_misfits.py — South Pole observation-vs-model fit at the frozen r8 MAP.

Runs the engine forward at the r8 MAP and plots, for each observable actually
on the tape (density, layer gradient d(age)/dz, borehole T, and one panel per
ApRES velocity block; plus depth-age when FIRN_AGE_BLOCK=1 restores it), the
observations with their 1-sigma errors against the model's kernel predictions
(the exact operators the objective uses), plus a combined standardized-residual
panel. The panel grid and letters follow the blocks present. This is the
paper's fit-quality figure.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/plot_misfits.py
"""
from __future__ import annotations
import json, math, string
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mcolors

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
    elif lab.startswith("v"): unit = "w (m yr⁻¹)"
    else: unit = lab
    return d, o, s, p, unit

blocks = {b["label"]: b for b in r["obs"]}
# the "age" block is deleted by default (redundant with dage; FIRN_AGE_BLOCK=1
# restores it), and velocity is per-site now -- so take whatever is present.
order = [l for l in ["rho", "age", "dage", "T"] if l in blocks] + \
        sorted(l for l in blocks if l.startswith("v"))
# per-site velocity blocks are labelled v_<site>; they all key on "v"
def base(lab): return "v" if lab.startswith("v") else lab
titles = {"rho": "density (SP19)", "age": "depth–age (SP19)",
          "dage": "layer gradient d(age)/dz", "T": "borehole temperature",
          "v": "vertical velocity (ApRES)"}
def title(lab):
    t = titles[base(lab)]
    return f"{t} — {lab[2:]}" if lab.startswith("v_") else t
C_O, C_M, C_R = "#111827", "#2563EB", "#EA580C"

# grid and panel letters follow the blocks actually present (+1 residual panel),
# so deleting or adding a block never leaves a hole or a mislabelled panel
NPAN = len(order) + 1
NC = min(3, NPAN); NR = math.ceil(NPAN/NC)
fig, AX = plt.subplots(NR, NC, figsize=(5*NC, 4.5*NR), squeeze=False)
axes = list(AX.flat)
for i, (ax, lab) in enumerate(zip(axes, order)):
    d, o, s, p, unit = to_display(blocks[lab])
    ax.errorbar(o, d, xerr=s, fmt="o", ms=3.5, color=C_O, elinewidth=0.7,
                capsize=0, alpha=0.75, label="observations ±1σ")
    oo = np.argsort(d)
    ax.plot(p[oo], d[oo], "-", color=C_M, lw=2, label=f"model ({MAP_NAME})")
    ax.invert_yaxis(); ax.set_ylim(132, 0)
    ax.set_xlabel(unit); ax.set_ylabel("depth (m)")
    rms = r["diag"]["rms_"+lab]
    ax.set_title(f"({string.ascii_lowercase[i]}) {title(lab)} — rms {rms:.2f}σ", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# final panel: all standardized residuals vs depth
ax = axes[len(order)]
for j in range(NPAN, len(axes)): axes[j].set_visible(False)
cols = {"rho": "#2563EB", "age": "#EA580C", "dage": "#059669", "T": "#7C3AED", "v": "#B45309"}
# every per-site velocity block gets its own legend entry, so it must get its own
# color: a ramp off the base hue keeps them readable as one family
VLABS = [l for l in order if base(l) == "v"]
def color(lab):
    if base(lab) != "v" or len(VLABS) < 2:
        return cols[base(lab)]
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(cols["v"]))
    f = VLABS.index(lab)/(len(VLABS) - 1)
    return mcolors.hsv_to_rgb((h, s*(1.0 - 0.6*f), min(1.0, v*(0.72 + 0.52*f))))
for lab in order:
    d, o, s, p, _ = to_display(blocks[lab])
    ax.plot((p-o)/s, d, "o", ms=3.5, color=color(lab), alpha=0.75, label=lab)
ax.axvline(0, color="k", lw=0.8); ax.axvspan(-1, 1, color="gray", alpha=0.12, lw=0)
ax.invert_yaxis(); ax.set_ylim(132, 0); ax.set_xlim(-3.2, 3.2)
ax.set_xlabel("(model − obs) / σ"); ax.set_ylabel("depth (m)")
ax.set_title(f"({string.ascii_lowercase[len(order)]}) standardized residuals, "
             f"all observables", fontsize=10.5)
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

fig.suptitle(f"South Pole joint assimilation — observations vs the {MAP_NAME} MAP "
             f"(J = {r['J']:.1f})", fontsize=13)
fig.tight_layout()
out = FIGS/("sp_misfits.png" if MAP_NAME == "sp_joint_r8"
            else f"sp_misfits_{MAP_NAME}.png")
fig.savefig(out, dpi=140)
print(f"Saved {out}")
