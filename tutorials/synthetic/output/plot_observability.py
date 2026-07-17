"""plot_observability.py — the "what does each observable buy" matrix.

Reads the six obsv_<slug>.json results + obsv_truth_profiles.json and shows,
per observation subset, the information fraction recovered for each target
quantity:

    f = clip( (|prior - truth| - |recovered - truth|) / |prior - truth| , 0, 1 )

f≈1: the subset recovers the quantity (data-constrained). f≈0: recovered value
stays at the prior (no information). Quantities where prior == truth are shown
as recovered-error annotations instead (f undefined).

Multiplicative quantities are compared in log space; histories use knot-rms.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent  # the tutorial dir, not output/
R = HERE/"results"
FIGS = HERE/"figures"; FIGS.mkdir(exist_ok=True)
tp = json.load(open(R/"obsv_truth_profiles.json"))
TRUTH = tp["truth"]
Ty, Tt = np.array(tp["T_years"]), np.array(tp["T_truth"])
By, Bt = np.array(tp["b_years"]), np.array(tp["b_truth"])
LIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0,
           k_snow_scale=1.0, k_firn_scale=1.0, Q_base=0.0, s2_shape=1.0)
T_PRIOR = np.full(len(Ty), -51.5); B_PRIOR = np.full(len(By), 0.09)
Rgas, TS = 8.314, 222.0

SUBSETS = ["rho", "rho-v", "rho-age", "rho-age-dage", "rho-age-dage-v",
           "rho-age-dage-Tdeep", "rho-age-dage-Tdeep-Tsh",
           "rho-age-dage-Tdeep-Tsh-v", "Tdeep"]
PRETTY = {"rho": "ρ", "rho-v": "ρ+w", "rho-age": "ρ+age",
          "rho-age-dage": "ρ+age+λ", "rho-age-dage-v": "ρ+age+λ+w",
          "rho-age-dage-Tdeep": "+deep T", "rho-age-dage-Tdeep-Tsh": "all core",
          "rho-age-dage-Tdeep-Tsh-v": "full (+w)", "Tdeep": "T only"}

def rates(m, b=0.09):
    return (m["hl_k0"]*math.exp(-m["hl_Ea1"]/(Rgas*TS))*b,
            m["hl_k1"]*math.exp(-m["hl_Ea2"]/(Rgas*TS))*math.sqrt(b))

c0_t, c1_t = rates(TRUTH); c0_p, c1_p = rates(LIT)

def logerr(a, b): return abs(math.log(a/b))

QUANT = [
    ("stage-1 rate", lambda m: logerr(rates(m)[0], c0_t), logerr(c0_p, c0_t)),
    ("stage-2 rate", lambda m: logerr(rates(m)[1], c1_t), logerr(c1_p, c1_t)),
    ("s2 (deep shape)", lambda m: logerr(m["s2_shape"], TRUTH["s2_shape"]), logerr(LIT["s2_shape"], TRUTH["s2_shape"])),
    ("k_snow", lambda m: logerr(m["k_snow_scale"], TRUTH["k_snow_scale"]), logerr(LIT["k_snow_scale"], TRUTH["k_snow_scale"])),
    ("Q_base", lambda m: abs(m["Q_base"]-TRUTH["Q_base"]), abs(LIT["Q_base"]-TRUTH["Q_base"])),
    ("ezz (dyn. strain)", lambda m: abs(m.get("ezz_yr", 0.0)-TRUTH["ezz_yr"]),
     abs(0.0-TRUTH["ezz_yr"])),
]

def hist_metrics(r):
    Trec = np.array(r["T_knots"]); Brec = np.array(r["b_knots"])
    # recent-T contrast (1950-1600) and deep-time T(1000); b(t) knot rms
    cT = (np.interp(1950, r["knot_years"], Trec) - np.interp(1600, r["knot_years"], Trec))
    m = {}
    m["recent T contrast"] = (abs(cT - (Tt[4]-Tt[2])), abs(0.0 - (Tt[4]-Tt[2])))
    m["deep-time T(1000)"] = (abs(Trec[0]-Tt[0]), abs(T_PRIOR[0]-Tt[0]))
    m["accum b(t) shape"] = (float(np.sqrt(np.mean((Brec-Bt)**2))),
                             float(np.sqrt(np.mean((B_PRIOR-Bt)**2))))
    return m

rows = [q[0] for q in QUANT] + ["recent T contrast", "deep-time T(1000)", "accum b(t) shape"]
# rows whose prior error is too small to test recovery (prior ~ truth):
# f is ill-conditioned there — hatch them instead of coloring.
MIN_PERR = {"stage-1 rate": 0.05, "stage-2 rate": 0.05, "s2 (deep shape)": 0.05,
            "k_snow": 0.05, "Q_base": 0.01, "ezz (dyn. strain)": 1e-5}
# rows that are structurally realization-dominated: the observing system's
# precision for them is no better than the prior, so single-draw f would
# color pure noise. Hatch with the reason instead.
ROW_NOTE = {"ezz (dyn. strain)":
            "not constrained at this geometry — single-profile w slope se(ezz) ≈ 1.2×10⁻⁴/yr ≥ prior σ; cells would show the noise draw",
            "k_snow":
            "mean T/ρ profiles carry ~no k_snow information (cells = noise/trades; same in the ezz-free archive) — the SEASONAL T amplitude does"}
F = np.full((len(rows), len(SUBSETS)), np.nan)
UNTESTABLE = np.zeros(len(rows), bool)
for i, (nm, err_fn, perr) in enumerate(QUANT):
    if perr < MIN_PERR.get(nm, 1e-9): UNTESTABLE[i] = True
for j, s in enumerate(SUBSETS):
    p = R/f"obsv_{s}.json"
    if not p.exists(): print(f"(missing {p})"); continue
    r = json.load(open(p)); m = r["m_map"]
    for i, (nm, err_fn, perr) in enumerate(QUANT):
        if UNTESTABLE[i]: continue
        F[i, j] = (perr - err_fn(m))/perr   # raw: negative = WORSE than prior
    hm = hist_metrics(r)
    for k, nm in enumerate(rows[len(QUANT):]):
        e, pe = hm[nm]
        if pe > 1e-9: F[len(QUANT)+k, j] = (pe-e)/pe

F_show = F.copy()
for i, nm in enumerate(rows):
    if nm in ROW_NOTE: F_show[i, :] = np.nan   # hatched rows: no cells underneath
fig, ax = plt.subplots(figsize=(11.8, 6.2))
im = ax.imshow(np.clip(F_show, 0, 1), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(SUBSETS))); ax.set_xticklabels([PRETTY[s] for s in SUBSETS], fontsize=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
for i in range(len(rows)):
    note = ROW_NOTE.get(rows[i])
    if (UNTESTABLE[i] if i < len(UNTESTABLE) else False) or note:
        ax.axhspan(i-0.5, i+0.5, color="gray", alpha=0.25, lw=0)
        ax.text(len(SUBSETS)/2-0.5, i,
                note or "prior ≈ truth (not testable in this OSSE)",
                ha="center", va="center", fontsize=8, color="#374151", style="italic")
        continue
    for j in range(len(SUBSETS)):
        if np.isfinite(F[i, j]):
            neg = F[i, j] < -0.005   # worse than prior: annotate in red
            ax.text(j, i, f"{F[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if F[i, j] > 0.6 else ("#B91C1C" if neg else "#111827"))
ax.set_title("OSSE observability: information fraction recovered per observation subset\n"
             "(1 = fully recovered from the prior toward truth; 0 = prior-bound)",
             fontsize=11)
fig.colorbar(im, ax=ax, label="information fraction f", shrink=0.85)
fig.tight_layout()
# console dump of the raw matrix for the record
print("f matrix (raw; negative = worse than prior):")
print(f"{'':22s}" + "".join(f"{PRETTY[s]:>12s}" for s in SUBSETS))
for i, nm in enumerate(rows):
    tag = " [hatched]" if (nm in ROW_NOTE or (i < len(UNTESTABLE) and UNTESTABLE[i])) else ""
    print(f"{nm:22s}" + "".join((f"{F[i,j]:12.2f}" if np.isfinite(F[i,j]) else f"{'—':>12s}")
                                for j in range(len(SUBSETS))) + tag)
out = FIGS/"observability_matrix.png"
fig.savefig(out, dpi=140); print(f"Saved {out}")
