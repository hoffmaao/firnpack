"""sp_obs_error_analysis.py — empirical observation-error correlation for the SP datasets.

Pure data analysis (no model): estimate the vertical autocorrelation of the
high-frequency (non-physical-profile) content of each profile dataset, and
convert it into an effective sample size for the assimilation subsampling:

    N_eff = N / (1 + 2 * sum_k rho(k*dz_sample))     (Bartlett)

This bounds the honest chi^2 weight between the two extremes used so far:
per-dataset 1/N (sigma inflated sqrt(N), current joint J) and fully
independent points (sigma as given). Age errors are cumulative (layer
counting), not stationary — recommendation there is to assimilate the
DERIVATIVE d(age)/dz (annual-layer thickness) instead; we quantify its
decorrelation too.

Output: results/sp_obs_error_analysis.{json,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

PROC = Path(__file__).parent.parent.parent / "processed"
OUT = Path(__file__).parent.parent.parent / "results"
H0 = 130.0

def smooth_anom(z, v, win_m):
    """Anomaly about a running-median smooth on a uniform grid; returns grid dz, anomaly."""
    zu = np.arange(z.min(), z.max(), max(np.median(np.diff(np.unique(z))), 0.05))
    vu = np.interp(zu, z, v)
    n = max(int(win_m / (zu[1]-zu[0])) | 1, 3)
    sm = pd.Series(vu).rolling(n, center=True, min_periods=1).median().values
    return zu[1]-zu[0], vu - sm

def acf(a, maxlag):
    a = a - a.mean(); c0 = np.dot(a, a)
    return np.array([np.dot(a[:len(a)-k], a[k:])/c0 for k in range(maxlag)])

def neff(N, dz_sample, dz_grid, rho):
    """Effective sample size for samples spaced dz_sample given grid-lag ACF rho."""
    s = 0.0
    k = 1
    while True:
        idx = int(round(k*dz_sample/dz_grid))
        if idx >= len(rho) or rho[idx] < 0.02: break
        s += rho[idx]; k += 1
    return N / (1.0 + 2.0*s)

results = {}

# ---- density ----
d = pd.read_csv(PROC/"sp19_density.csv").query("depth_m<=@H0")
z, v = d.depth_m.values, d.rho_kgm3.values*1000.0
dzg, an = smooth_anom(z, v, win_m=5.0)
rho_acf = acf(an, int(15.0/dzg))
L = dzg*np.argmax(rho_acf < 1/np.e) if (rho_acf < 1/np.e).any() else 15.0
N_assim, dz_assim = 45, H0/45
ne = neff(N_assim, dz_assim, dzg, rho_acf)
results["density"] = dict(n_raw=len(z), noise_sd=float(an.std()), L_corr_m=float(L),
                          N_assim=N_assim, dz_assim_m=dz_assim, N_eff=float(ne))
print(f"density: raw n={len(z)}, anomaly sd={an.std():.1f} kg/m3, L_corr={L:.2f} m, "
      f"N_eff({N_assim} pts @ {dz_assim:.1f} m) = {ne:.1f}")

# ---- borehole temperature ----
t = pd.read_csv(PROC/"spicecore_borehole_T.csv").query("depth_m<=@H0")
z, v = t.depth_m.values, t.T_C.values
dzg, an = smooth_anom(z, v, win_m=20.0)   # physical profile is very smooth; 20 m window
rho_acf_T = acf(an, min(int(40.0/dzg), len(an)-2))
LT = dzg*np.argmax(rho_acf_T < 1/np.e) if (rho_acf_T < 1/np.e).any() else 40.0
NT, dzT = 40, H0/40
neT = neff(NT, dzT, dzg, rho_acf_T)
results["temperature"] = dict(n_raw=len(z), noise_sd=float(an.std()), L_corr_m=float(LT),
                              N_assim=NT, dz_assim_m=dzT, N_eff=float(neT))
print(f"temperature: raw n={len(z)}, anomaly sd={an.std():.3f} C, L_corr={LT:.1f} m, "
      f"N_eff({NT} pts @ {dzT:.1f} m) = {neT:.1f}")

# ---- age: work in layer-thickness (derivative) space ----
a = pd.read_csv(PROC/"sp19_depth_age.csv"); a["age_yr"] = 2015.0-a.year_CE
a = a.query("depth_m<=@H0 and age_yr>=0").sort_values("depth_m")
za, va = a.depth_m.values, a.age_yr.values
dadz = np.gradient(va, za)                      # yr/m — inverse layer thickness
dzg, an = smooth_anom(za[1:-1], dadz[1:-1], win_m=5.0)
rho_acf_a = acf(an, min(int(15.0/dzg), len(an)-2))
La = dzg*np.argmax(rho_acf_a < 1/np.e) if (rho_acf_a < 1/np.e).any() else 15.0
Na, dza = 40, H0/40
nea = neff(Na, dza, dzg, rho_acf_a)
results["age_gradient"] = dict(n_raw=len(za), noise_sd=float(an.std()), L_corr_m=float(La),
                               N_assim=Na, dz_assim_m=dza, N_eff=float(nea),
                               note="cumulative age errors are non-stationary; "
                                    "assimilate d(age)/dz to decorrelate")
print(f"age gradient: L_corr={La:.2f} m, N_eff({Na} pts) = {nea:.1f}")

json.dump(results, open(OUT/"sp_obs_error_analysis.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, AX = plt.subplots(1, 3, figsize=(14, 4))
def panel(ax, name, rho_arr, dz_grid, rr):
    lags = np.arange(len(rho_arr))*dz_grid
    ax.plot(lags, rho_arr, "C0-")
    ax.axhline(1/np.e, color="k", ls=":", lw=1)
    ax.axvline(rr["L_corr_m"], color="C3", ls="--", lw=1, label=f"L={rr['L_corr_m']:.1f} m")
    ax.axvline(rr["dz_assim_m"], color="C2", ls="--", lw=1, label=f"assim Δz={rr['dz_assim_m']:.1f} m")
    ax.set_title(f"{name}: N_eff={rr['N_eff']:.0f}/{rr['N_assim']}")
    ax.set_xlabel("lag (m)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
dz1 = smooth_anom(d.depth_m.values, d.rho_kgm3.values*1000.0, 5.0)[0]
panel(AX[0], "density", rho_acf, dz1, results["density"])
dz2 = smooth_anom(t.depth_m.values, t.T_C.values, 20.0)[0]
panel(AX[1], "temperature", rho_acf_T, dz2, results["temperature"])
dz3 = smooth_anom(za[1:-1], dadz[1:-1], 5.0)[0]
panel(AX[2], "age gradient", rho_acf_a, dz3, results["age_gradient"])
AX[0].set_ylabel("autocorrelation")
fig.suptitle("Observation-noise vertical autocorrelation → effective sample sizes", fontsize=12)
fig.tight_layout()
fig.savefig(OUT/"sp_obs_error_analysis.png", dpi=120)
print(f"Saved {OUT/'sp_obs_error_analysis.json'} and .png")
