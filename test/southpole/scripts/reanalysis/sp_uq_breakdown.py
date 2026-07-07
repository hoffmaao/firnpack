"""sp_uq_breakdown.py — decompose the reanalysis-headline uncertainty by posterior mode.

Which parameter combinations carry sigma(h')? For each Hessian eigenmode u_i with
eigenvalue lam_i, the variance contribution to q = s.x is (s.u_i)^2 / lam_i.
Also: posterior sigma of the recovered warming amplitude Tk4 - Tk0, and of the
rate-relevant recent cooling Tk5 - Tk4.

numpy only. Reads results/sp_uq_hessian.json + sp_uq_sensitivity.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).parent.parent.parent / "results"
Hd = json.load(open(OUT / "sp_uq_hessian.json"))
Sd = json.load(open(OUT / "sp_uq_sensitivity.json"))
names = Hd["names"]; N = len(names)
H = np.array(Hd["H_sym"]); S = np.array(Sd["S"]); q0 = np.array(Sd["q_base"])
lam, U = np.linalg.eigh(H)
Sigma = (U / lam) @ U.T

s_h = S[0]  # dh'(2015)/dx
var_tot = float(s_h @ Sigma @ s_h)
print(f"sigma(h') total = {np.sqrt(var_tot)*100:.1f} cm\n")
print("Per-eigenmode contribution to var(h'):")
contrib = (s_h @ U) ** 2 / lam
idx = np.argsort(contrib)[::-1]
for i in idx[:6]:
    v = U[:, i]; vmax = np.abs(v).max()
    load = "  ".join(f"{names[j]}:{v[j]/vmax:+.2f}" for j in range(N) if abs(v[j]) > 0.25 * vmax)
    print(f"  lam={lam[i]:9.3e}  contrib={np.sqrt(contrib[i])*100:7.1f} cm  ({100*contrib[i]/var_tot:4.1f}%)  [{load}]")

def lincomb_sigma(coef):
    return float(np.sqrt(coef @ Sigma @ coef))

for label, a, b in [("warming amplitude Tk4-Tk0", "Tk4", "Tk0"),
                    ("recent cooling   Tk5-Tk4", "Tk5", "Tk4")]:
    c = np.zeros(N); c[names.index(a)] = 1.0; c[names.index(b)] = -1.0
    x = np.array(Hd["x_star"])
    val = x[names.index(a)] - x[names.index(b)]
    print(f"\n{label} = {val:+.3f} +/- {lincomb_sigma(c):.3f} C")

# effective constraint on the stage-1/2 rates at the SP mean temperature
# d log(rate1) = d log k0 - dEa1/(R T); in internal coords Ea is log-space:
# d log(rate1) = dlk0 - (Ea1/(R T)) dlEa1
R_GAS, T_MEAN = 8.314, 273.15 - 45.2
x = np.array(Hd["x_star"])
import math
for st, kn, En in [("stage-1 rate", "hl_k0", "hl_Ea1"), ("stage-2 rate", "hl_k1", "hl_Ea2")]:
    Ea = math.exp(x[names.index(En)])
    c = np.zeros(N); c[names.index(kn)] = 1.0; c[names.index(En)] = -Ea / (R_GAS * T_MEAN)
    print(f"log {st} @ {T_MEAN-273.15:.1f}C : sigma = {lincomb_sigma(c):.3f} (multiplicative x/ {math.exp(lincomb_sigma(c)):.2f})")
