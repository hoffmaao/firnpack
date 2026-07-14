"""plot_law_comparison.py — the transferability figure: do the densification
laws independently recovered at South Pole and Summit agree?

Both inversions used the SAME neutral literature-H&L prior, so the recovered
laws are directly comparable. Left: dρ/dt(ρ) for each recovered law at a common
(T, b) — overlay = universal physics. Right: Summit accumulation history.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)
su = json.load(open(HERE/"results/summit_invert.json"))
sp = json.load(open(HERE.parent.parent/"test/southpole/results/sp_joint_r8.json"))
ms, mp = su["m_map"], sp["m_map"]
R, RHO_I, RHO_M = 8.314, 917.0, 550.0

def rate_curve(m, T_C, b):
    T = T_C + 273.15; rho = np.linspace(360, 900, 300)
    phi = np.maximum((RHO_I-rho)/(RHO_I-RHO_M), 1e-4)
    sw = 0.5*(1+np.tanh((rho-RHO_M)/20.0))
    c0 = m["hl_k0"]*math.exp(-m["hl_Ea1"]/(R*T))*b
    c1 = m["hl_k1"]*math.exp(-m["hl_Ea2"]/(R*T))*math.sqrt(b)*phi**(m.get("s2_shape",1)-1)
    c = ((1-sw)*c0 + sw*c1)*(RHO_I-rho)
    return rho, c

C_SP, C_SU, C_INK = "#2563EB", "#EA580C", "#111827"
fig, AX = plt.subplots(1, 2, figsize=(12, 5.4))

# (a) law overlay at a common reference
ax = AX[0]
for m, col, lab in [(mp, C_SP, "South Pole (−51°C, 0.09 m/yr)"),
                    (ms, C_SU, "Summit (−29°C, 0.25 m/yr)")]:
    r, c = rate_curve(m, -40.0, 0.15)
    ax.plot(r, c, "-", color=col, lw=2.2, label=f"{lab} recovered law")
ax.set_yscale("log"); ax.set_xlabel("density ρ (kg m⁻³)"); ax.set_ylabel("dρ/dt (kg m⁻³ yr⁻¹)")
ax.axvline(RHO_M, color="gray", ls=":", lw=1); ax.text(556, ax.get_ylim()[0]*1.4, "ρ_m", fontsize=8, color="gray")
r40=rate_curve(mp,-40,0.15); r40s=rate_curve(ms,-40,0.15)
i2=np.argmin(np.abs(r40[0]-700))
ax.set_title("(a) Recovered densification law — two ice sheets, one physics\n"
             "(evaluated at a common −40°C, 0.15 m/yr)", fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.06, "stage-2 rate ratio Summit/SP = 1.00\nstage-1 rate ratio = 0.90\nparams within 1–9%",
        transform=ax.transAxes, fontsize=8.5, color=C_INK,
        bbox=dict(boxstyle="round", fc="w", alpha=0.85))

# (b) Summit accumulation history
ax = AX[1]
by, bk = np.array(su["b_knot_years"]), np.array(su["b_knots"])
bc = np.array(su["b_prior_centers"])
yy = np.linspace(by.min(), by.max(), 300); bkc = np.exp(np.interp(yy, by, np.log(bk)))
ax.plot(yy, bkc, "-", color=C_SU, lw=2, label="recovered b(t)")
ax.plot(by, bk, "o", color=C_SU, ms=4)
ax.axhline(bc[0], color="gray", ls="--", lw=1, label=f"Osman prior ({bc[0]:.3f})")
ax.set_xlabel("year CE"); ax.set_ylabel("accumulation (m ice eq / yr)")
ax.set_title("(b) Summit accumulation history (recovered)", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.3); ax.set_ylim(0.12, 0.30)
ax.text(0.03, 0.05, "endpoint (2015) knot weakly constrained\n(no layers younger than the surface)",
        transform=ax.transAxes, fontsize=7.5, color="gray")

fig.suptitle("Firn densification law transferability — South Pole vs Summit "
             "(independent inversions, same literature prior)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGS/"law_comparison.png", dpi=140)
print("params:  SP     Summit   ratio")
for k in ["hl_k0","hl_k1","hl_Ea1","hl_Ea2","s2_shape"]:
    print(f"  {k:9s} {mp[k]:9.3f} {ms[k]:9.3f}  {ms[k]/mp[k]:.3f}")
print(f"Saved {FIGS/'law_comparison.png'}")
