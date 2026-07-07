#!/usr/bin/env python3
"""
southpole_apres_crim_diagnostic.py
===================================
Diagnostic plots for ApRES apparent-range change at South Pole.

Data:
  * Stacked complex profiles (Hills 2020, USAP-DC 601503)
    - 2018_19/  : epoch 1 (~Jan 2019)
    - 2019_20/  : epoch 2 (~Dec 2019)
  * Pre-computed VV (Hills 2020) for comparison

Physics:
  Apparent range (in ice-equivalent metres) to a reflector at depth z_R:
      R(z_R) = ∫₀^{z_R}  n(ρ(z)) / n_ice  dz
             = z_R  +  [(n_ice-1)/n_ice] / ρ_ice  ×  ∫₀^{z_R} ρ(z) dz

  where CRIM gives  n(ρ) = 1 + (n_ice - 1) * ρ / ρ_ice.

  Interferometric phase between epochs Δφ = angle(S₂ · S₁*) gives:
      ΔR = Δφ · λ_eff / (4π),   λ_eff = c / (n_ice · f_c)  ≈ 0.563 m

Figures produced:
  diag_apres_crim_amplitude.png   – raw stacked-power profiles, both epochs
  diag_apres_crim_deltaR.png      – apparent range change, phase vs Hills VV
  diag_apres_crim_physics.png     – CRIM n(ρ) curve + ΔR decomposition sketch
"""

import re
import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data" / "usap_dc" / "601503"
STACKED   = DATA_ROOT / "stacked_profiles" / "stacked_profiles"
VV_DIR    = DATA_ROOT / "vertical_velocities" / "vertical_velocities"
OUT_DIR   = Path(__file__).parent.parent / "results"
OUT_DIR.mkdir(exist_ok=True)

# ─── Physical constants ───────────────────────────────────────────────────
C_LIGHT  = 3.0e8        # m/s
F_C      = 300e6        # BAS ApRES centre frequency (Hz)
N_ICE    = 1.775        # refractive index of pure ice
RHO_ICE  = 917.0        # kg/m³
LAM_C    = C_LIGHT / F_C            # free-space wavelength = 1 m
LAM_EFF  = LAM_C / N_ICE            # ice-equivalent wavelength ≈ 0.5634 m

# ─── Sites with VV files in both epochs ──────────────────────────────────
SITES = ["x5n2","x5s2","x8n0","x11n0","x11n2","x11n6",
         "x11s10","x11s2","x17n2","x17s2"]
# Extract approximate distance-from-pole (km) from site name for colouring
def site_dist_km(name):
    import re
    m = re.match(r'x(\d+)', name)
    return int(m.group(1)) if m else 0

COH_MIN   = 0.3   # minimum coherence for Hills VV
FIRN_MAX  = 150.0 # m — depth range shown in firn plots
SMOOTH_M  = 20.0  # smoothing window for interferometric phase (m)

# ─── I/O helpers ─────────────────────────────────────────────────────────
def load_stacked_profile(fpath):
    """Return (decimal_day_matlab, range_m, complex_return) from a stacked profile file."""
    with open(fpath) as f:
        line1 = f.readline()
        f.readline()                    # column-header line
        decimal_day = float(line1.split()[1])
        lines = f.readlines()
    N = len(lines)
    ranges  = np.zeros(N)
    returns = np.zeros(N, dtype=complex)
    pat = re.compile(r'\(([^)]+)\)')
    for i, line in enumerate(lines):
        m = pat.findall(line)
        ranges[i]  = complex(m[0]).real     # Range col: always real
        returns[i] = complex(m[1])
    return decimal_day, ranges, returns


def matlab_day_to_date(d):
    """Convert MATLAB serial date to Python datetime.date (epoch = Jan 1, 0000)."""
    return datetime.date.fromordinal(int(d) - 366)


def load_hills_vv(site, coh_min=COH_MIN):
    """Return (range_m, vv_m_yr, coherence) from Hills pre-processed VV file."""
    fpath = VV_DIR / f"{site}_VV.txt"
    if not fpath.exists():
        return None, None, None
    d = np.loadtxt(fpath, skiprows=1)
    r, coh, vv = d[:, 0], d[:, 1], d[:, 3]
    vv = vv.copy()
    vv[coh < coh_min] = np.nan
    return r, vv, coh


# ─── Core computation: interferometric ΔR ────────────────────────────────
def compute_delta_R(s1, s2, range_m, smooth_m=SMOOTH_M):
    """
    Compute per-bin interferometric apparent-range change (m) with range smoothing.

    Parameters
    ----------
    s1, s2    : complex arrays (same length)
    range_m   : range axis (m)
    smooth_m  : smoothing window half-width in metres

    Returns
    -------
    dR_ice    : ΔR in ice-equivalent metres (positive = range increasing)
    coh       : local coherence in smoothing window
    """
    dr = range_m[1] - range_m[0]
    w  = max(1, int(round(smooth_m / dr)))

    # Complex cross-product
    interf = s2 * np.conj(s1)

    # Smooth in range (complex averaging to preserve phase)
    interf_sm  = uniform_filter1d(interf.real, w) + 1j * uniform_filter1d(interf.imag, w)
    amp1_sm    = uniform_filter1d(np.abs(s1) ** 2, w)
    amp2_sm    = uniform_filter1d(np.abs(s2) ** 2, w)

    phase  = np.angle(interf_sm)
    dR_ice = phase * LAM_EFF / (4.0 * np.pi)

    # Coherence from smoothed product
    denom = np.sqrt(amp1_sm * amp2_sm) + 1e-40
    coh   = np.abs(interf_sm) / denom

    return dR_ice, coh


# ─── Load all sites ──────────────────────────────────────────────────────
print("Loading stacked profiles …")

site_data = {}
dt_days   = None

for site in SITES:
    f1 = STACKED / "2018_19" / f"{site}_VV.txt"
    f2 = STACKED / "2019_20" / f"{site}_VV.txt"
    if not (f1.exists() and f2.exists()):
        print(f"  skipping {site} (file missing)")
        continue

    dd1, r1, s1 = load_stacked_profile(f1)
    dd2, r2, s2 = load_stacked_profile(f2)

    if dt_days is None:
        dt_days = dd2 - dd1
        date1   = matlab_day_to_date(dd1)
        date2   = matlab_day_to_date(dd2)
        print(f"  Epoch 1: {date1}  Epoch 2: {date2}  Δt = {dt_days:.1f} d")

    assert np.allclose(r1, r2), f"Range mismatch for {site}"
    dR_ice, coh_interf = compute_delta_R(s1, s2, r1, SMOOTH_M)

    r_hills, vv_hills, coh_hills = load_hills_vv(site)
    site_data[site] = dict(
        range_m    = r1,
        s1=s1, s2=s2,
        dR_ice     = dR_ice,
        coh_interf = coh_interf,
        r_hills    = r_hills,
        vv_hills   = vv_hills,
        coh_hills  = coh_hills,
    )
    print(f"  {site}: loaded ({len(r1)} bins)")

dt_yr = dt_days / 365.25

# ─── Figure 1: Raw amplitude profiles ──────────────────────────────────
print("\nFigure 1: amplitude profiles …")

sites_plot = [s for s in ["x5n2", "x8n0", "x11n0", "x17n2"] if s in site_data]
fig, axes = plt.subplots(1, len(sites_plot), figsize=(4 * len(sites_plot), 6),
                         sharey=True, constrained_layout=True)
fig.suptitle("ApRES stacked-power profiles — South Pole firn column", fontsize=13)

for ax, site in zip(axes, sites_plot):
    d     = site_data[site]
    r     = d["range_m"]
    mask  = r <= FIRN_MAX
    amp1  = 20 * np.log10(np.abs(d["s1"])[mask] + 1e-12)
    amp2  = 20 * np.log10(np.abs(d["s2"])[mask] + 1e-12)
    ax.plot(amp1, r[mask], lw=0.6, color="steelblue", label=str(date1))
    ax.plot(amp2, r[mask], lw=0.6, color="firebrick", alpha=0.7, label=str(date2))
    ax.invert_yaxis()
    ax.set_xlabel("Amplitude (dB)")
    ax.set_title(site)
    ax.grid(True, lw=0.3, alpha=0.5)
    if ax is axes[0]:
        ax.set_ylabel("Range (m, ice-equiv.)")
    ax.legend(fontsize=8)

fig.savefig(OUT_DIR / "diag_apres_crim_amplitude.png", dpi=150)
plt.close(fig)
print("  saved diag_apres_crim_amplitude.png")

# ─── Figure 2: ΔR comparison (phase vs Hills VV×Δt) ───────────────────
print("Figure 2: ΔR comparison …")

all_sites = sorted(site_data.keys(), key=site_dist_km)
dists     = [site_dist_km(s) for s in all_sites]
cmap      = cm.plasma
norm      = plt.Normalize(vmin=min(dists), vmax=max(dists))

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
fig2.suptitle(
    f"ApRES apparent-range change  |  {date1} → {date2}  (Δt = {dt_yr:.2f} yr)",
    fontsize=13)

ax_phase, ax_vv, ax_scatter = axes2

for site in all_sites:
    d      = site_data[site]
    r      = d["range_m"]
    dist   = site_dist_km(site)
    color  = cmap(norm(dist))
    mask_f = r <= FIRN_MAX
    coh_ok = d["coh_interf"] >= COH_MIN

    # Panel A: ΔR from interferometry vs range
    r_plot = r[mask_f & coh_ok]
    dR_plot = d["dR_ice"][mask_f & coh_ok]
    # Convert to annual rate for comparison
    ax_phase.plot(dR_plot / dt_yr * 100,  # cm/yr
                  r_plot,
                  lw=0.8, color=color, label=site, alpha=0.8)

    # Panel B: Hills VV directly
    if d["r_hills"] is not None:
        rh   = d["r_hills"]
        vh   = d["vv_hills"]
        mf_h = rh <= FIRN_MAX
        ax_vv.plot(vh[mf_h] * 100,   # cm/yr
                   rh[mf_h],
                   lw=0.8, color=color, alpha=0.8)

    # Panel C: scatter ΔR/yr (phase) vs Hills VV, at Hills VV range bins
    if d["r_hills"] is not None:
        rh   = d["r_hills"]
        vh   = d["vv_hills"]
        mf_h = (rh > 5) & (rh <= FIRN_MAX) & np.isfinite(vh)
        # Interpolate phase-derived rate to Hills range grid
        dR_rate_interp = np.interp(rh[mf_h], r, d["dR_ice"]) / dt_yr
        ax_scatter.scatter(vh[mf_h] * 100,
                           dR_rate_interp * 100,
                           s=6, color=color, alpha=0.6)

ax_phase.axvline(0, color="k", lw=0.5)
ax_phase.set_xlabel("ΔR / year (cm yr⁻¹,  from phase)")
ax_phase.set_ylabel("Range (m, ice-equiv.)")
ax_phase.invert_yaxis()
ax_phase.set_title("Interferometric ΔR rate")
ax_phase.grid(True, lw=0.3, alpha=0.5)
# colourbar proxy
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig2.colorbar(sm, ax=ax_phase, label="Distance from pole (km)")

ax_vv.axvline(0, color="k", lw=0.5)
ax_vv.set_xlabel("Vertical velocity (cm yr⁻¹,  Hills 2020)")
ax_vv.invert_yaxis()
ax_vv.set_title("Hills VV (pre-computed)")
ax_vv.grid(True, lw=0.3, alpha=0.5)
fig2.colorbar(sm, ax=ax_vv, label="Distance from pole (km)")

# 1:1 line for scatter
all_x = []
for site in all_sites:
    d = site_data[site]
    if d["r_hills"] is None:
        continue
    rh = d["r_hills"]
    vh = d["vv_hills"]
    mf = (rh > 5) & (rh <= FIRN_MAX) & np.isfinite(vh)
    all_x.extend(vh[mf] * 100)
if all_x:
    lims = (min(all_x), max(all_x))
    ax_scatter.plot(lims, lims, "k--", lw=1, label="1:1")
ax_scatter.set_xlabel("Hills VV (cm yr⁻¹)")
ax_scatter.set_ylabel("Phase-derived ΔR/yr (cm yr⁻¹)")
ax_scatter.set_title("Comparison: phase vs Hills VV")
ax_scatter.legend(fontsize=9)
ax_scatter.set_aspect("equal", adjustable="datalim")
ax_scatter.grid(True, lw=0.3, alpha=0.5)
fig2.colorbar(sm, ax=ax_scatter, label="Distance from pole (km)")

fig2.savefig(OUT_DIR / "diag_apres_crim_deltaR.png", dpi=150)
plt.close(fig2)
print("  saved diag_apres_crim_deltaR.png")

# ─── Figure 3: CRIM physics ─────────────────────────────────────────────
print("Figure 3: CRIM physics …")

fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
fig3.suptitle("CRIM: density → refractive index → apparent range", fontsize=13)

ax_n, ax_R, ax_dRdz = axes3

# Panel A: CRIM n(ρ) curve
rho_arr = np.linspace(0, RHO_ICE, 400)
n_arr   = 1.0 + (N_ICE - 1.0) * rho_arr / RHO_ICE
ax_n.plot(n_arr, rho_arr, "steelblue", lw=2)
ax_n.axhline(RHO_ICE, color="gray", lw=0.8, ls="--", label=f"Pure ice ρ={RHO_ICE:.0f}")
ax_n.set_xlabel("Refractive index  n(ρ)")
ax_n.set_ylabel("Density  ρ (kg m⁻³)")
ax_n.set_title("CRIM relationship")
ax_n.legend(fontsize=9)
ax_n.grid(True, lw=0.3, alpha=0.5)
ax_n.text(1.05, 50, r"$n(\rho)=1+\frac{n_\mathrm{ice}-1}{\rho_\mathrm{ice}}\rho$",
          fontsize=11)

# Panel B: Apparent range R(z_R) vs true depth z_R for a model density profile
# Simple Herron-Langway-like exponential profile for illustration
z_arr   = np.linspace(0, 130, 500)
rho_s   = 350.0    # surface density kg/m³
rho_fcn = RHO_ICE - (RHO_ICE - rho_s) * np.exp(-z_arr / 30.0)  # e-folding ~30 m

n_fcn   = 1.0 + (N_ICE - 1.0) * rho_fcn / RHO_ICE
R_app   = np.cumsum(n_fcn / N_ICE * np.gradient(z_arr))  # ice-equiv range (trapz)

ax_R.plot(z_arr, z_arr, "k--", lw=1, label="True depth (z)")
ax_R.plot(R_app, z_arr, "steelblue", lw=2, label="Apparent range R(z)")
ax_R.set_xlabel("Depth / apparent range (m)")
ax_R.set_ylabel("True depth z (m)")
ax_R.invert_yaxis()
ax_R.legend(fontsize=9)
ax_R.set_title("Apparent range vs true depth")
ax_R.grid(True, lw=0.3, alpha=0.5)
extra = R_app - z_arr
ax_R.fill_betweenx(z_arr, z_arr, R_app,
                   alpha=0.15, color="steelblue",
                   label=f"Extra path (max {extra[-1]:.1f} m)")
ax_R.legend(fontsize=9)

# Panel C: dΔR/dz per unit density change — sensitivity kernel
# ΔR(z_R) = ∫₀^{z_R} [n(ρ₂) - n(ρ₁)] / n_ice dz
#          = (n_ice-1)/(n_ice * ρ_ice) ∫₀^{z_R} Δρ dz
# Sensitivity: dΔR/dΔρ(z') = (n_ice-1)/(n_ice * ρ_ice) × [z_R - z']  (box function)
# Show this kernel for three reflector depths
drho_factor = (N_ICE - 1.0) / (N_ICE * RHO_ICE)  # (m³/kg) sensitivity per metre
z_ax = np.linspace(0, 130, 400)
for z_R, col in [(30, "steelblue"), (70, "seagreen"), (110, "firebrick")]:
    kern = np.where(z_ax <= z_R, drho_factor, 0.0)  # m³/kg — sensitivity per unit depth
    ax_dRdz.plot(kern * 1e4,  # scale to 1e-4 m per (kg/m³)
                 z_ax, color=col, lw=2, label=f"z_R = {z_R} m")
ax_dRdz.set_xlabel(r"$\partial R / \partial \rho \;[\times 10^{-4}\;\mathrm{m}\,(\mathrm{kg\,m^{-3}})^{-1}]$")
ax_dRdz.set_ylabel("Depth z  (m)")
ax_dRdz.invert_yaxis()
ax_dRdz.set_title("Sensitivity kernel  ∂R(z_R)/∂ρ(z)")
ax_dRdz.legend(fontsize=9)
ax_dRdz.grid(True, lw=0.3, alpha=0.5)
ax_dRdz.text(0.55, 0.03,
             r"$\frac{\partial R(z_R)}{\partial \rho(z)}=\frac{n_\mathrm{ice}-1}{n_\mathrm{ice}\,\rho_\mathrm{ice}}\,\mathbf{1}_{z\leq z_R}$",
             transform=ax_dRdz.transAxes, fontsize=10)

fig3.savefig(OUT_DIR / "diag_apres_crim_physics.png", dpi=150)
plt.close(fig3)
print("  saved diag_apres_crim_physics.png")

print("\nDone.")
