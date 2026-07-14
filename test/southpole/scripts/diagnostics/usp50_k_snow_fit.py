"""usp50_k_snow_fit.py — near-surface conductivity from USP50 seasonal damping.

Breaks the r5 k_snow_scale <-> recent-T degeneracy with an INDEPENDENT
measurement: the USP50 instrumented hole (data/usap_dc/601525, 23 thermistors
0-40 m, 6-hourly 2017-01..2018-12) records the annual temperature wave whose
amplitude decay and phase lag with depth are set by the conductivity profile.

Method
  1. Per sensor: least-squares harmonic fit T(t) = c0 + c1 t + annual +
     semiannual -> complex annual amplitude A e^{i phi}.
  2. Forward model: complex harmonic heat conduction (k(z) T')' = i w rho c T
     solved as a tridiagonal BVP on z in [z_top, 45 m], k(z) = Calonne-2019
     with k_firn_scale = 0.987 (r5, solid) and k_snow_scale = s (the unknown);
     rho(z) from the USP50 density core + pit. Advection is negligible
     (w ~ 0.23 m/yr << omega*skin ~ 16 m/yr).
  3. Pin the solution to the complex amplitude at the shallowest RELIABLE
     sensor (>= 1.0 m; the top sensors see radiation/ventilation and burial);
     fit s by minimizing joint ln-amplitude + phase residuals at deeper
     sensors with A > 0.05 C. Leave-one-out spread -> sigma_s.

Caveats (documented, small): sensor burial (~0.23 m/yr snow) treated by
using mid-record effective depths (z + 0.23 m; sensitivity reported);
pure conduction (no ventilation below ~1 m); USP50 is a nearby site, we
transfer the k(rho) LAW not the site.

Output: results/usp50_k_snow_fit.{json,png}
Run: /home/andrew/venv-firedrake-2026/bin/python usp50_k_snow_fit.py  (numpy only)
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent.parent
OUT = BASE / "results"
U = BASE / "data/usap_dc"
K_FIRN_SCALE = 0.987          # r5 MAP (rock solid)
RHO_I, C_I = 917.0, 2009.0    # match firnpack.constants heat capacity
T_MEAN_K = 222.25             # -50.9 C: USP50 measured firn mean (corrected datum)
BURIAL = 0.23                 # m/yr snow-surface rise -> mid-record depth shift
A_MIN, Z_PIN = 0.05, 1.0      # usable amplitude floor; shallowest pinned sensor
OMEGA = 2.0 * math.pi         # 1/yr

# ---- load thermistor records (file is transposed; depth in last column) ----
raw = pd.read_csv(U / "601525/USP50_firn_temperatures_170109-181224.csv", index_col=0)
tstamp = pd.to_datetime(raw.loc["TIMESTAMP"].drop("depth"))
t_yr = (tstamp - tstamp.iloc[0]).dt.total_seconds().values / (365.25 * 86400.0)
sensors = [r for r in raw.index if r != "TIMESTAMP"]
z0 = raw["depth"][sensors].astype(float)
T = raw.loc[sensors].drop(columns="depth").astype(float)

# ---- density profile rho(z): pit (shallow) + 106 m core, extended ----
dens = pd.read_csv(U / "601680/usapdc_601680/USP50_density_106.csv")
zd, rd = dens.iloc[:, 0].values.astype(float), dens.iloc[:, 1].values.astype(float) * 1000.0
try:
    pit = pd.read_csv(U / "601680/usapdc_601680/USP50_density_pit.csv")
    zp, rp = pit.iloc[:, 0].values.astype(float), pit.iloc[:, 1].values.astype(float)
    if rp.max() < 10: rp = rp * 1000.0
    zd = np.concatenate([zp, zd]); rd = np.concatenate([rp, rd])
    o = np.argsort(zd); zd, rd = zd[o], rd[o]
except Exception as e:
    print(f"(pit density not used: {e})")
def rho_of(z):
    return np.interp(z, zd, rd, left=rd[0], right=rd[-1])

# ---- Calonne-2019 k(rho) at site temperature ----
def k_calonne(rho, s_snow):
    k_sn = 0.024 - 1.23e-4 * rho + 2.5e-6 * rho**2
    k_fi = 2.107 + 3.618e-3 * (rho - RHO_I)
    th = 1.0 / (1.0 + np.exp(-2.0 * 0.02 * (rho - 450.0)))
    FT = math.exp(-5.7e-3 * (T_MEAN_K - 270.15))
    return FT * ((1.0 - th) * s_snow * k_sn + th * K_FIRN_SCALE * k_fi)

# ---- per-sensor harmonic fit ----
rows = []
for s in sensors:
    y = T.loc[s].values
    ok = np.isfinite(y) & (y > -80) & (y < 0)
    if ok.sum() < 1000: continue
    tt, yy = t_yr[ok], y[ok]
    X = np.column_stack([np.ones_like(tt), tt,
                         np.cos(OMEGA*tt), np.sin(OMEGA*tt),
                         np.cos(2*OMEGA*tt), np.sin(2*OMEGA*tt)])
    c, *_ = np.linalg.lstsq(X, yy, rcond=None)
    A = math.hypot(c[2], c[3]); ph = math.atan2(-c[3], c[2])  # T ~ A cos(wt + ph)
    rows.append(dict(sensor=s, z=float(z0[s]), zeff=float(z0[s]) + BURIAL,
                     mean=float(c[0]), A=A, phase=ph, n=int(ok.sum())))
H = pd.DataFrame(rows).sort_values("zeff").reset_index(drop=True)
print(H[["z", "zeff", "mean", "A", "phase"]].to_string(index=False,
      float_format=lambda v: f"{v:8.3f}"))

# complex observed amplitude C_obs = A * exp(i*phase)
H["C"] = H.A * np.exp(1j * H.phase)
use = H[(H.zeff >= Z_PIN) & (H.A >= A_MIN)].reset_index(drop=True)
pin = use.iloc[0]; fitpts = use.iloc[1:]
print(f"\npin at z={pin.zeff:.2f} m (A={pin.A:.2f}); fitting {len(fitpts)} sensors "
      f"to z={fitpts.zeff.max():.1f} m")

# ---- harmonic BVP: (k Th')' = i w rho c Th on [z_pin, 45], Th(z_pin)=C_pin ----
zg = np.arange(float(pin.zeff), 45.0 + 1e-9, 0.05)
def model_profile(s_snow):
    n = len(zg); dz = zg[1] - zg[0]
    k_e = k_calonne(rho_of(0.5 * (zg[:-1] + zg[1:])), s_snow)      # edges
    iwrc = 1j * OMEGA / (365.25*86400.0) * rho_of(zg) * C_I        # SI omega
    lo = np.zeros(n, complex); di = np.zeros(n, complex); up = np.zeros(n, complex)
    b = np.zeros(n, complex)
    di[0] = 1.0; b[0] = pin.C
    for i in range(1, n-1):
        lo[i] = k_e[i-1]/dz**2; up[i] = k_e[i]/dz**2
        di[i] = -(k_e[i-1]+k_e[i])/dz**2 - iwrc[i]
    # radiating bottom: Th' = -Th/d_loc
    d_loc = math.sqrt(2*k_calonne(rho_of(zg[-1]), s_snow) /
                      (OMEGA/(365.25*86400.0)*rho_of(zg[-1])*C_I))
    lo[-1] = -1.0/dz; di[-1] = 1.0/dz + 1.0/d_loc
    # Thomas solve
    cp = np.zeros(n, complex); bp = np.zeros(n, complex)
    cp[0] = up[0]/di[0]; bp[0] = b[0]/di[0]
    for i in range(1, n):
        m = di[i] - lo[i]*cp[i-1]
        cp[i] = up[i]/m if i < n-1 else 0.0
        bp[i] = (b[i] - lo[i]*bp[i-1])/m
    x = np.zeros(n, complex); x[-1] = bp[-1]
    for i in range(n-2, -1, -1): x[i] = bp[i] - cp[i]*x[i+1]
    return x

def resid(s_snow, pts):
    prof = model_profile(s_snow)
    r = []
    for _, p in pts.iterrows():
        Cm = np.interp(p.zeff, zg, prof.real) + 1j*np.interp(p.zeff, zg, prof.imag)
        r.append(math.log(abs(Cm)/p.A))
        dph = np.angle(Cm/p.C)
        r.append(dph)
    return np.array(r)

def fit_s(pts):
    ss = np.linspace(0.2, 2.5, 47)
    cost = [float(np.sum(resid(s, pts)**2)) for s in ss]
    i = int(np.argmin(cost))
    lo, hi = ss[max(i-1, 0)], ss[min(i+1, len(ss)-1)]
    for _ in range(40):
        m1, m2 = lo + (hi-lo)/3, hi - (hi-lo)/3
        if np.sum(resid(m1, pts)**2) < np.sum(resid(m2, pts)**2): hi = m2
        else: lo = m1
    return 0.5*(lo+hi)

s_fit = fit_s(fitpts)
loo = [fit_s(fitpts.drop(i)) for i in fitpts.index]
s_sig = float(np.std(loo, ddof=1))
# burial sensitivity: refit with install depths
use0 = use.copy(); use0["zeff"] = use0["z"]
s_nob = fit_s(use0.iloc[1:])
rms = float(np.sqrt(np.mean(resid(s_fit, fitpts)**2)))
print(f"\nk_snow_scale = {s_fit:.3f} +/- {s_sig:.3f} (leave-one-out)"
      f"   [no-burial-corr: {s_nob:.3f}]   joint lnA+phase rms {rms:.3f}")
r5v = 1.576
print(f"r5 MAP had {r5v} (degenerate direction); leg-1 had 0.527")

json.dump(dict(s_snow=s_fit, sigma_loo=s_sig, s_no_burial=s_nob,
               rms=rms, k_firn_scale_assumed=K_FIRN_SCALE,
               pin_z=float(pin.zeff), n_fit=len(fitpts),
               sensors=H[["z", "zeff", "mean", "A", "phase"]].to_dict("records")),
          open(OUT/"usp50_k_snow_fit.json", "w"), indent=1)

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C_POST, C_PRIOR, C_INK = "#2563EB", "#6B7280", "#111827"
fig, AX = plt.subplots(1, 3, figsize=(13, 4.6))
prof_fit = model_profile(s_fit); prof_1 = model_profile(1.0)
ax = AX[0]
ax.semilogx(H.A, H.zeff, "o", ms=5, mfc="none", mec=C_INK, label="USP50 sensors")
ax.semilogx(np.abs(prof_fit), zg, "-", color=C_POST, lw=2,
            label=f"Calonne, s_snow={s_fit:.2f}")
ax.semilogx(np.abs(prof_1), zg, "--", color=C_PRIOR, lw=1.5, label="s_snow=1")
ax.invert_yaxis(); ax.set_ylim(22, 0)
ax.set_xlabel("annual amplitude (°C)"); ax.set_ylabel("depth (m)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("(a) amplitude decay")
ax = AX[1]
ph_u = np.unwrap([p for p in use.phase])
ax.plot(ph_u - ph_u[0], use.zeff, "o", ms=5, mfc="none", mec=C_INK)
ax.plot(np.unwrap(np.angle(prof_fit)) - np.angle(prof_fit[0]), zg, "-",
        color=C_POST, lw=2)
ax.plot(np.unwrap(np.angle(prof_1)) - np.angle(prof_1[0]), zg, "--",
        color=C_PRIOR, lw=1.5)
ax.invert_yaxis(); ax.set_ylim(22, 0)
ax.set_xlabel("phase lag rel. pin (rad)"); ax.set_ylabel("depth (m)")
ax.grid(alpha=0.3); ax.set_title("(b) phase lag")
ax = AX[2]
rr = np.linspace(340, 700, 200)
ax.plot(rr, k_calonne(rr, s_fit), "-", color=C_POST, lw=2,
        label=f"fit s_snow={s_fit:.2f}±{s_sig:.2f}")
ax.plot(rr, k_calonne(rr, 1.0), "--", color=C_PRIOR, lw=1.5, label="published (s=1)")
ax.plot(rr, k_calonne(rr, 0.527), ":", color=C_PRIOR, lw=1.5, label="r5 leg-1 (0.53)")
ax.set_xlabel("density (kg m⁻³)"); ax.set_ylabel("k (W m⁻¹ K⁻¹)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_title("(c) implied k(ρ) at −45 °C")
fig.suptitle("USP50 seasonal damping → near-surface conductivity (independent of the inversion)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT/"usp50_k_snow_fit.png", dpi=130)
print(f"Saved {OUT/'usp50_k_snow_fit.json'} and .png")
