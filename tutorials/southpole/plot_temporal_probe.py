"""plot_temporal_probe.py — does denser temporal sampling help at South Pole?

Three-panel verdict figure for the dense-knot probe (r8 coarse 13T+15b vs
dense 20T+34b on the identical clean configuration):
  (a) b(t): both MAPs against the MODEL-FREE raw-layer apparent accumulation
      (b_app = lambda*rho/rho_i) — the arbiter: structure the raw layers don't
      show is null-space artifact, not signal.
  (b) T(t): both recovered histories.
  (c) fit rms per observable + J for both — what the extra 26 controls bought.

Run: PYTHONPATH=src OMP_NUM_THREADS=1 <venv> tutorials/southpole/plot_temporal_probe.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)
R8 = json.load(open(HERE.parent.parent/"test/southpole/results/sp_joint_r8.json"))
DN = json.load(open(HERE/"results/sp_engine_dense.json"))
RHO_I = 917.0

# ---- engine forward at each MAP (exact rms through the objective operators) ----
from firnpack.inverse import assimilate
def cfg_for(knots_file):
    ns = {"__file__": str(HERE/"run.py"), "__name__": "cfgbuild"}
    if knots_file: os.environ["FIRN_KNOTS"] = knots_file
    else: os.environ.pop("FIRN_KNOTS", None)
    src = open(HERE/"run.py").read().split("warm = json.load")[0]
    exec(src, ns)
    return ns["cfg"]
rms = {}
for lab, kf, warm in [("r8 (13T+15b)", "", R8), ("dense (20T+34b)", "knots_dense.json", DN)]:
    r = assimilate(cfg_for(kf), mode="forward", warm=warm, verbose=False)
    rms[lab] = {k[4:]: v for k, v in r["diag"].items() if k.startswith("rms_")}
    rms[lab]["J"] = r["J"]
    print(f"{lab}: J={r['J']:.2f} " + " ".join(f"{k}={v:.2f}" for k, v in rms[lab].items() if k != "J"))

# ---- raw-layer apparent accumulation ----
a = pd.read_csv(HERE/"data/sp19_depth_age.csv")
a = a[(a.depth_m <= 130) & (a.year_CE <= 2015)].sort_values("depth_m")
age = 2015.0 - a.year_CE.values; o = np.argsort(age)
age_s, z_s = age[o], a.depth_m.values[o]
lam = np.gradient(z_s, age_s)
dens = pd.read_csv(HERE/"data/sp19_density.csv")
rho = np.interp(z_s, dens.depth_m.values, dens.rho_kgm3.values*1000.0)
ezz = abs(float(R8["m_map"].get("ezz_yr", 0.0)))
b_app = lam*rho/RHO_I/np.exp(-ezz*age_s)
dep = 2015.0 - age_s
def runmed(x, w=5): return np.array([np.median(x[max(0,i-w//2):i+w//2+1]) for i in range(len(x))])
b_sm = runmed(b_app, 5)

C_R8, C_DN, C_RAW = "#111827", "#2563EB", "#9CA3AF"
fig, AX = plt.subplots(1, 3, figsize=(15.5, 4.9))

ax = AX[0]
oo = np.argsort(dep)
ax.plot(dep[oo], b_sm[oo], "-", color=C_RAW, lw=1.6, label="raw layers b_app (5-yr median)")
for lab, r, c, ls in [("r8 coarse", R8, C_R8, "-"), ("dense", DN, C_DN, "--")]:
    yy = np.linspace(1000, 2015, 400)
    bb = np.exp(np.interp(yy, r["b_knot_years"], np.log(r["b_knots"])))
    ax.plot(yy, bb, ls, color=c, lw=2, label=f"{lab} MAP b(t)")
    ax.plot(r["b_knot_years"], r["b_knots"], "o", color=c, ms=3)
ax.axvline(1950, color="#EA580C", ls=":", lw=1)
ax.annotate("dense re-grows the 1950 bump;\nraw layers are locally LOW there\n→ null-space artifact, not signal",
            xy=(1950, 0.106), xytext=(1520, 0.112), fontsize=8.5, color="#EA580C",
            arrowprops=dict(arrowstyle="->", color="#EA580C", lw=1))
ax.set_xlim(1000, 2020); ax.set_ylim(0.055, 0.125)
ax.set_xlabel("year CE"); ax.set_ylabel("accumulation (m ice/yr)")
ax.set_title("(a) accumulation: MAPs vs model-free raw layers", fontsize=10.5)
ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

ax = AX[1]
for lab, r, c, ls in [("r8 coarse", R8, C_R8, "-"), ("dense", DN, C_DN, "--")]:
    ax.plot(r["knot_years"], r["T_knots"], ls, color=c, lw=1.8, marker="o", ms=3.5, label=lab)
ax.set_xlabel("year CE"); ax.set_ylabel("surface T (°C)")
ax.set_title("(b) temperature history", fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = AX[2]
obs_order = ["rho", "age", "dage", "T", "v"]
x = np.arange(len(obs_order)); w = 0.38
for i, (lab, c) in enumerate([("r8 (13T+15b)", C_R8), ("dense (20T+34b)", C_DN)]):
    ax.bar(x + (i-0.5)*w, [rms[lab][k] for k in obs_order], w*0.92, color=c, alpha=0.85,
           label=f"{lab}: J={rms[lab]['J']:.1f}")
ax.axhline(1.0, color="gray", ls=":", lw=1)
ax.set_xticks(x); ax.set_xticklabels(obs_order)
ax.set_ylabel("misfit rms (σ)"); ax.set_ylim(0, 1.35)
ax.set_title("(c) fit quality: +26 controls buy ΔJ = "
             f"{rms['dense (20T+34b)']['J']-rms['r8 (13T+15b)']['J']:+.2f}", fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

fig.suptitle("Temporal-sampling probe (South Pole, clean r8 configuration): "
             "denser T/b knots refill the null space, not the data", fontsize=12)
fig.tight_layout()
out = FIGS/"temporal_probe.png"
fig.savefig(out, dpi=140); print(f"Saved {out}")
