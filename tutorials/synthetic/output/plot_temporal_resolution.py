"""plot_temporal_resolution.py — OSSE arbiter for the knot-density question.

Truth WITH decadal structure, inverted with coarse (22-ctrl) vs dense (87-ctrl)
layouts from identical observations. Shows truth vs both recoveries for T(t)
and b(t), with band-split rms in the panel titles.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not output/; R = HERE/"results"
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)
tr = json.load(open(R/"tr_truth.json"))
Ty, Tt = np.array(tr["T_years"]), np.array(tr["T_truth"])
By, Bt = np.array(tr["B_years"]), np.array(tr["B_truth"])
yy = np.arange(1000, 2016, 5.0)
Tt_g = np.interp(yy, Ty, Tt); Bt_g = np.exp(np.interp(yy, By, np.log(Bt)))
def smooth(v, w=50):
    k = int(w/5); k += (k+1) % 2
    return np.convolve(v, np.ones(k)/k, mode="same")
sel = (yy >= 1050) & (yy <= 1990)

C_T, C_C, C_D = "#111827", "#2563EB", "#EA580C"
fig, AX = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
stats = {}
for case, col, ls in [("coarse", C_C, "-"), ("dense", C_D, "--")]:
    r = json.load(open(R/f"tr_{case}.json"))
    Tr = np.interp(yy, r["knot_years"], r["T_knots"])
    Br = np.exp(np.interp(yy, r["b_knot_years"], np.log(r["b_knots"])))
    stats[case] = dict(
        T=np.sqrt(np.mean((Tr-Tt_g)[sel]**2)),
        Tm=np.sqrt(np.mean((smooth(Tr)-smooth(Tt_g))[sel]**2)),
        b=1e3*np.sqrt(np.mean((Br-Bt_g)[sel]**2)),
        bm=1e3*np.sqrt(np.mean((smooth(Br)-smooth(Bt_g))[sel]**2)),
        n=r["n_ctrl"], Tr=Tr, Br=Br)

ax = AX[0]
ax.plot(yy, Tt_g, "-", color=C_T, lw=2.2, label="truth (has decadal structure)")
for case, col, ls in [("coarse", C_C, "-"), ("dense", C_D, "--")]:
    s = stats[case]
    ax.plot(yy, s["Tr"], ls, color=col, lw=1.7,
            label=f"{case} ({s['n']} ctrls): rms {s['T']:.2f} °C (multidec {s['Tm']:.2f})")
ax.axhline(-51.5, color="gray", lw=0.8, ls=":"); ax.text(1005, -51.45, "prior", fontsize=8, color="gray")
ax.set_ylabel("surface T (°C)")
ax.set_title("(a) temperature history: dense knots DEGRADE the multidecadal recovery", fontsize=11)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

ax = AX[1]
ax.plot(yy, Bt_g, "-", color=C_T, lw=2.2, label="truth")
for case, col, ls in [("coarse", C_C, "-"), ("dense", C_D, "--")]:
    s = stats[case]
    ax.plot(yy, s["Br"], ls, color=col, lw=1.7,
            label=f"{case}: rms {s['b']:.1f} mm/yr (multidec {s['bm']:.1f})")
ax.axhline(0.09, color="gray", lw=0.8, ls=":")
ax.set_ylabel("accumulation (m ice/yr)"); ax.set_xlabel("year CE")
ax.set_title("(b) accumulation history: decadal structure unrecoverable; dense adds spurious variance", fontsize=11)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

fig.suptitle("OSSE temporal-resolution twin — known decadally-structured truth, identical observations,\n"
             "coarse vs dense knot layouts: extra temporal freedom harms, not helps", fontsize=12)
fig.tight_layout()
out = FIGS/"temporal_resolution_osse.png"
fig.savefig(out, dpi=140); print(f"Saved {out}")
for c in stats: print(c, {k: round(v,3) for k,v in stats[c].items() if not hasattr(v,'__len__')})
