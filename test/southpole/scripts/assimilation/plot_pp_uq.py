"""plot_pp_uq.py — figures for the per-point Laplace verdict.

Reads results/{sp_joint_pp,sp_uq_result_pp,sp_uq_hessian_pp,sp_uq_sensitivity_pp}.json
(numpy/matplotlib only, no Firedrake) and renders:

  results/sp_pp_uq_verdict.png — 4 panels:
    (a) scalar controls: posterior 68% vs prior (prior-sigma units)
    (b) T(t): MAP + posterior band vs prior band (contrasts annotated)
    (c) b(t): MAP + band, Buizert/ERA5 prior centers, stake range, the 1950 seam
    (d) attribution significance (q/sigma) + var(h') source split

  results/sp_pp_law_envelope.png — dρ/dt(ρ) at −45.2 °C, b=0.08:
    posterior 68% envelope (joint draws, correlations included), MAP,
    MAP with s2=1, literature H&L.

Run: /home/andrew/venv-firedrake-2026/bin/python plot_pp_uq.py
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent.parent.parent / "results"
PROC = Path(__file__).parent.parent.parent / "processed"
PP = json.load(open(OUT/"sp_joint_pp.json"))
UQ = json.load(open(OUT/"sp_uq_result_pp.json"))
HD = json.load(open(OUT/"sp_uq_hessian_pp.json"))
SD = json.load(open(OUT/"sp_uq_sensitivity_pp.json"))

names = UQ["names"]; N = len(names)
x = np.array(UQ["x_star"]); Sig = np.array(UQ["Sigma"]); sig = np.array(UQ["sigma_marginal"])
SCAL = ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor", "Q_base", "ezz_yr", "s2_shape"]
N_SCAL = len(SCAL)
LOGS = {"hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "k_factor", "s2_shape"}
INIT = dict(hl_k0=10.79, hl_k1=570.7, hl_Ea1=10432.0, hl_Ea2=21875.0,
            k_factor=0.5, Q_base=0.0, ezz_yr=0.0, s2_shape=1.0)
PRIOR_SIG = dict(hl_k0=1.0, hl_k1=1.0, hl_Ea1=0.5, hl_Ea2=0.5, k_factor=1.0,
                 Q_base=0.05, ezz_yr=1.0e-4, s2_shape=0.3)
T_SIG = 0.6
KY = np.array(PP["knot_years"]); TK = np.array(PP["T_knots"]); NT = len(KY)
BY = np.array(PP["b_knot_years"]); BK = np.array(PP["b_knots"]); NB = len(BY)
BCTR = np.array(PP["b_prior_centers"]); SIG_LOGB = float(PP["sig_logB"])
_T6Y = np.array([1000., 1300., 1550., 1750., 1900., 2015.])
_T6V = np.array([-45.50, -45.45, -45.35, -45.20, -45.10, -45.00])
T_CTR = np.interp(KY, _T6Y, _T6V)
sig_T = sig[N_SCAL:N_SCAL+NT]; sig_lb = sig[N_SCAL+NT:]

C_POST, C_PRIOR, C_FLAG, C_INK = "#2563EB", "#6B7280", "#EA580C", "#111827"

def contrast_sigma(i, j):
    return math.sqrt(Sig[i, i] + Sig[j, j] - 2*Sig[i, j])

fig = plt.figure(figsize=(13.5, 9.8))
gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24,
                      left=0.065, right=0.975, top=0.93, bottom=0.06)

# ------------------------------------------------------------------ (a)
ax = fig.add_subplot(gs[0, 0])
rows = list(reversed(range(N_SCAL)))
ax.axvspan(-1, 1, color=C_PRIOR, alpha=0.13, lw=0)
ax.axvline(0, color=C_PRIOR, lw=0.8)
lab_val = {
    "hl_k0": f"{math.exp(x[0]):.1f} ×/{math.exp(sig[0]):.2f}",
    "hl_k1": f"{math.exp(x[1]):.0f} ×/{math.exp(sig[1]):.2f}",
    "hl_Ea1": f"{math.exp(x[2])/1e3:.2f} kJ/mol ×/{math.exp(sig[2]):.2f}",
    "hl_Ea2": f"{math.exp(x[3])/1e3:.2f} kJ/mol ×/{math.exp(sig[3]):.2f}",
    "k_factor": f"{math.exp(x[4]):.3f} ×/{math.exp(sig[4]):.2f}",
    "Q_base": f"{x[5]*1e3:+.1f}±{sig[5]*1e3:.1f} mW/m²",
    "ezz_yr": f"{x[6]*1e5:+.1f}±{sig[6]*1e5:.1f} ×10⁻⁵/yr",
    "s2_shape": f"{math.exp(x[7]):.2f} ×/{math.exp(sig[7]):.2f}",
}
pretty = dict(hl_k0="k0", hl_k1="k1", hl_Ea1="Ea1", hl_Ea2="Ea2",
              k_factor="k_factor", Q_base="Q_base", ezz_yr="ezz",
              s2_shape="s2 shape")
for r, nm in zip(rows, SCAL):
    i = names.index(nm)
    ctr = math.log(INIT[nm]) if nm in LOGS else INIT[nm]
    z = (x[i]-ctr)/PRIOR_SIG[nm]; w = sig[i]/PRIOR_SIG[nm]
    flag = abs(x[i]-ctr)/max(sig[i], 1e-30) >= 2.0
    col = C_FLAG if flag else C_POST
    ax.plot([z-w, z+w], [r, r], color=col, lw=2.4, solid_capstyle="round")
    ax.plot([z], [r], "o", color=col, ms=6)
    ax.text(3.05, r, lab_val[nm], va="center", ha="right", fontsize=8.5,
            color=C_INK)
ax.set_yticks(rows); ax.set_yticklabels([pretty[n] for n in SCAL], fontsize=9)
ax.set_xlim(-2.7, 3.15); ax.set_ylim(-0.7, N_SCAL-0.3)
ax.set_xlabel("(MAP − prior center) / prior σ", fontsize=9)
ax.set_title("(a) Scalar controls: posterior 68% vs prior", fontsize=11)
ax.text(-1, N_SCAL-0.52, "prior ±1σ", color=C_PRIOR, fontsize=8, ha="left")
ax.text(0.02, 0.03, "orange = MAP ≥2 posterior σ from prior center",
        transform=ax.transAxes, fontsize=8, color=C_FLAG)
ax.grid(alpha=0.25, axis="x")

# ------------------------------------------------------------------ (b)
ax = fig.add_subplot(gs[0, 1])
yy = np.linspace(1000, 2015, 400)
t_map = np.interp(yy, KY, TK); t_ctr = np.interp(yy, KY, T_CTR)
t_sig = np.interp(yy, KY, sig_T)
ax.fill_between(yy, t_ctr-T_SIG, t_ctr+T_SIG, color=C_PRIOR, alpha=0.16, lw=0,
                label="prior ±1σ")
ax.plot(yy, t_ctr, color=C_PRIOR, lw=1.2, ls="--")
ax.fill_between(yy, t_map-t_sig, t_map+t_sig, color=C_POST, alpha=0.20, lw=0,
                label="posterior ±1σ")
ax.plot(yy, t_map, color=C_POST, lw=2)
ax.plot(KY, TK, "o", color=C_POST, ms=3.5)
# References, NOT priors — and different physical quantities (see legend):
# Buizert is d18O-derived condensation/cloud T (project offset +5.8 C to
# surface level; variability transfer is state-dependent, so shape is
# indicative only). ERA5 is 2 m surface T with its own plateau biases.
import pandas as pd
_bz = pd.read_csv(PROC/"buizert2021_spice_temp.csv").query("year_CE>=1000")
ax.plot(_bz.year_CE, _bz.temp+5.8, color=C_INK, lw=1.1, ls="-.", alpha=0.75,
        label="Buizert δ18O cloud T +5.8 °C (ref.)")
_e = pd.read_csv(PROC/"era5_monthly_point.csv")
_ea = (_e.groupby("year")["t2m_K"].mean()-273.15)
_ea = _ea[(_ea.index >= 1957) & (_ea.index <= 2015)]   # instrument era only
_e5 = _ea.rolling(10, center=True, min_periods=5).mean()
ax.plot(_e5.index, _e5.values, color=C_PRIOR, lw=1.3, alpha=0.95,
        label="ERA5 t2m, 10-yr mean (ref.)")
i5, i1 = names.index("Tk5"), names.index("Tk1")
amp, s_amp = TK[5]-TK[1], contrast_sigma(i5, i1)
i14, i11 = names.index("Tk14"), names.index("Tk11")
mod, s_mod = TK[14]-TK[11], contrast_sigma(i14, i11)
ax.annotate(f"amplitude 1150→1750:\n+{amp:.2f} ± {s_amp:.2f} °C  ({amp/s_amp:.1f}σ)",
            xy=(1750, TK[5]), xytext=(1090, -44.38), fontsize=8.5, color=C_INK,
            arrowprops=dict(arrowstyle="-", color=C_INK, lw=0.7, alpha=0.6))
ax.annotate(f"1965→2015: +{mod:.2f} ± {s_mod:.2f} °C  ({mod/s_mod:.1f}σ)",
            xy=(2000, TK[14]), xytext=(1430, -45.92), fontsize=8.5, color=C_INK,
            arrowprops=dict(arrowstyle="-", color=C_INK, lw=0.7, alpha=0.6))
ax.set_title("(b) Surface-T history: posterior band ≈ prior band", fontsize=11)
ax.set_ylabel("°C"); ax.set_xlabel("year CE", fontsize=9)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.92)
ax.set_xlim(1000, 2020); ax.set_ylim(-46.55, -44.15); ax.grid(alpha=0.25)

# ------------------------------------------------------------------ (c)
ax = fig.add_subplot(gs[1, 0])
lb_map = np.log(BK)
b_lo = np.exp(np.interp(yy, BY, lb_map - sig_lb))
b_hi = np.exp(np.interp(yy, BY, lb_map + sig_lb))
b_map = np.exp(np.interp(yy, BY, lb_map))
ax.axhspan(0.085, 0.093, color=C_PRIOR, alpha=0.10, lw=0)
ax.text(1015, 0.0935, "station-era stake estimates (approx.)", fontsize=7.5,
        color=C_PRIOR)
ax.fill_between(yy, b_lo, b_hi, color=C_POST, alpha=0.20, lw=0, label="posterior ±1σ")
ax.plot(yy, b_map, color=C_POST, lw=2, label="MAP b(t)")
mb = BY <= 1950
ax.plot(BY[mb], BCTR[mb], "o", mfc="none", mec=C_PRIOR, ms=5, mew=1.2,
        label="prior: Buizert (≤1950)")
ax.plot(BY[~mb], BCTR[~mb], "s", mfc="none", mec=C_PRIOR, ms=5, mew=1.2,
        label="prior: ERA5 (>1950)")
i50, i62 = names.index("b1950"), names.index("b1962")
seam = math.exp(x[i50]-x[i62]); s_seam = contrast_sigma(i50, i62)
zs = (x[i50]-x[i62])/s_seam
ax.annotate("", xy=(1962, BK[15]), xytext=(1950, BK[14]),
            arrowprops=dict(arrowstyle="->", color=C_FLAG, lw=1.6))
ax.text(1560, 0.0625, f"datum seam: b(1950)/b(1962) = {seam:.2f}× ({zs:.1f}σ)\n"
        "Buizert↔ERA5 offset, not physics", fontsize=8.5, color=C_FLAG)
ax.set_title("(c) Accumulation history: posterior tracks the priors — and their seam",
             fontsize=11)
ax.set_ylabel("m ice eq / yr"); ax.set_xlabel("year CE", fontsize=9)
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.set_xlim(1000, 2020); ax.set_ylim(0.058, 0.125); ax.grid(alpha=0.25)

# ------------------------------------------------------------------ (d)
sub = gs[1, 1].subgridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.55)
ax = fig.add_subplot(sub[0])
q = UQ["headline"]
items = [("Δh′ total since 1000 CE", q["hprime_2015_m"], 1.0, "m"),
         ("dh/dt 1957–2015", q["rate_1957_2015_m_yr"], 1e3, "mm/yr"),
         ("ΔFAC (firn air) 2015", q["dfac_2015_m"], 100.0, "cm")]
ax.axvspan(-1, 1, color=C_PRIOR, alpha=0.13, lw=0)
ax.axvline(0, color=C_PRIOR, lw=0.8)
ylabs = []
for r, (lab, d, s, u) in enumerate(reversed(items)):
    z = d["value"]/d["sigma"]
    ax.plot([z-1, z+1], [r, r], color=C_POST, lw=2.4, solid_capstyle="round")
    ax.plot([z], [r], "o", color=C_POST, ms=6)
    ylabs.append(f"{lab}\n{d['value']*s:+.2f} ± {d['sigma']*s:.2f} {u}")
ax.set_yticks(range(len(items)))
ax.set_yticklabels(ylabs, fontsize=8.5)
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("value / posterior σ", fontsize=9)
ax.set_title("(d) Attribution headline: all consistent with zero", fontsize=11)
ax.grid(alpha=0.25, axis="x")

# var(h') split by control block, from eigenmodes of the scaled Hessian
axb = fig.add_subplot(sub[1])
scale = np.array(HD["scale"]); Hs = np.array(HD["H_sym"])*np.outer(scale, scale)
lam, U = np.linalg.eigh(Hs); lam = np.maximum(lam, 1e-8*lam.max())
Ss = np.array(SD["S"])[0]*scale
proj = Ss @ U; contrib = proj**2/lam
blocks = dict(physics=slice(0, N_SCAL), T=slice(N_SCAL, N_SCAL+NT),
              b=slice(N_SCAL+NT, N))
frac = {}
for m in range(N):
    u2 = U[:, m]**2
    key = max(blocks, key=lambda k: u2[blocks[k]].sum())
    frac[key] = frac.get(key, 0.0) + contrib[m]
tot = sum(frac.values()); left = 0.0
cols = dict(b=C_POST, T="#93B4F8", physics=C_PRIOR)
for k in ["b", "T", "physics"]:
    w = 100*frac.get(k, 0)/tot
    axb.barh([0], [w], left=left, height=0.5, color=cols[k], alpha=0.85)
    lab = {"b": "b-knot priors", "T": "T-knots", "physics": "physics"}[k]
    if w > 8:
        axb.text(left+w/2, 0, f"{lab}\n{w:.0f}%", ha="center", va="center",
                 fontsize=8, color="white" if k == "b" else C_INK)
    left += w
axb.set_xlim(0, 100); axb.set_yticks([]); axb.set_xticks([0, 25, 50, 75, 100])
axb.tick_params(labelsize=8)
rest = 100*(frac.get("T", 0)+frac.get("physics", 0))/tot
axb.set_title(f"where var(Δh′) comes from (eigenmode blocks; T + physics: {rest:.0f}%)",
              fontsize=9.5)

fig.suptitle("South Pole joint inversion — per-point (honest) Laplace posterior, "
             f"46 controls, J={PP['J']:.2f}", fontsize=12.5)
fig.savefig(OUT/"sp_pp_uq_verdict.png", dpi=140)
print(f"Saved {OUT/'sp_pp_uq_verdict.png'}")

# ================================================================== Fig 2
R, RHO_I, RHO_M, SPY = 8.314, 917.0, 550.0, 1.0
T_C, B0 = -45.2, 0.08
Tk = T_C + 273.15
rho = np.linspace(360.0, 870.0, 300)
phi = np.maximum((RHO_I-rho)/(RHO_I-RHO_M), 1e-4)
sw = 0.5*(1.0+np.tanh((rho-RHO_M)/20.0))

def hl_rate(k0, k1, e1, e2, s2):
    c0 = k0*math.exp(-e1/(R*Tk))*B0
    c1 = k1*math.exp(-e2/(R*Tk))*math.sqrt(B0)*phi**(s2-1.0)
    return ((1.0-sw)*c0 + sw*c1)*(RHO_I-rho)   # kg m^-3 yr^-1

idx = [names.index(n) for n in ["hl_k0", "hl_k1", "hl_Ea1", "hl_Ea2", "s2_shape"]]
mu = x[idx]; C = Sig[np.ix_(idx, idx)]
rng = np.random.default_rng(0)
draws = rng.multivariate_normal(mu, C, size=500)
rates = np.array([hl_rate(*np.exp(d)) for d in draws])
lo, hi = np.percentile(rates, [16, 84], axis=0)
r_map = hl_rate(*np.exp(mu))
r_map_s1 = hl_rate(*np.exp(mu[:4]), 1.0)
r_lit = hl_rate(11.0, 575.0, 10160.0, 21400.0, 1.0)

f2, ax = plt.subplots(figsize=(7.6, 5.6))
ax.axvspan(780, 870, color=C_PRIOR, alpha=0.08, lw=0)
ax.text(824, 0.062, "deep signal\nregion (ρ>780)", fontsize=8, color=C_PRIOR,
        ha="center")
ax.fill_between(rho, lo, hi, color=C_POST, alpha=0.22, lw=0,
                label="posterior 68% (joint draws)")
ax.plot(rho, r_map, color=C_POST, lw=2.2, label=f"MAP (s2={math.exp(mu[4]):.2f})")
ax.plot(rho, r_map_s1, color=C_POST, lw=1.4, ls=":", label="MAP params, s2=1")
ax.plot(rho, r_lit, color=C_INK, lw=1.4, ls="--", label="literature H&L")
ax.axvline(RHO_M, color=C_PRIOR, lw=1.0, ls=":")
ax.text(552, 0.055, "ρ_m", fontsize=8.5, color=C_PRIOR)
i830 = int(np.argmin(np.abs(rho-830.0)))
boost = 100.0*(r_map[i830]/r_map_s1[i830]-1.0)
ax.annotate(f"s2<1 sustains deep densification\n(+{boost:.0f}% at ρ=830 — what the\nρ>780 residual demanded)",
            xy=(830, math.sqrt(r_map[i830]*r_map_s1[i830])), xytext=(600, 0.115),
            fontsize=8.5, color=C_INK,
            arrowprops=dict(arrowstyle="->", color=C_INK, lw=0.9, alpha=0.7))
ax.set_yscale("log")
ax.set_xlabel("density ρ (kg m⁻³)")
ax.set_ylabel("dρ/dt (kg m⁻³ yr⁻¹)")
ax.set_title(f"Recovered densification law at {T_C} °C, b = {B0} m ice/yr\n"
             "posterior envelope includes the k↔Ea correlations", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
ax.grid(alpha=0.25, which="both")
f2.tight_layout()
f2.savefig(OUT/"sp_pp_law_envelope.png", dpi=140)
print(f"Saved {OUT/'sp_pp_law_envelope.png'}")
print("Done.")
